"""
UNet Model for 2D Medical Image Segmentation
with MOL-Conditional Modules + DINO + PointRend
using Pretrained CrossAttentionQueryExtractor (Learnable Query).

Architecture:
  - Module 1: LearnableQueryAdapter — Generates dynamic prototype vector from frozen
    CrossAttentionQueryExtractor -> cosine sim -> MOL-dependent temperature scaling -> adapter CNN
    -> feature injection to bottleneck.
    * Query module is frozen, no alignment_loss (already pre-trained).
  - Module 2: MOLFiLMPointHead — FiLM-modulated PointRend head
  - Module 3: DINO-Guided Sampling — uncertainty + DINO sim-weighted point sampling

Training Phases:
  Phase 1 (Warmup): PointRend OFF → L_seg + L_aux
  Phase 2 (Refinement): PointRend ON → L_seg + L_aux + L_point (uniform)

How to train:
  cd /path/to/TISC

  CUDA_VISIBLE_DEVICES=0 python train_slice.py \\
      --model unet_all_mol_learnable_query \\
      --fold 0 --epochs 200 --warm_up 20 \\
      --learnable_query_ckpt results/module/learnable_query/fold0/checkpoints/best_query_module.pth
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .module_learnable_query import CrossAttentionQueryExtractor


# ============================================================================
# Utility Functions for Point Sampling
# ============================================================================

def point_sample(input, point_coords, **kwargs):
    """
    A wrapper around torch.nn.functional.grid_sample to support 3D point_coords tensors.
    Assumes `point_coords` to lie inside [0, 1] x [0, 1] square.

    Args:
        input (Tensor): (N, C, H, W) feature map.
        point_coords (Tensor): (N, P, 2) or (N, Hgrid, Wgrid, 2) normalized [0,1] coords.

    Returns:
        output (Tensor): (N, C, P) or (N, C, Hgrid, Wgrid) sampled features.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


@torch.no_grad()
def sampling_points(mask, N, k=3, beta=0.75, training=True):
    """
    Standard uncertainty-based point sampling (from PointRend).

    Args:
        mask: [B, C, H, W] logits
        N: number of points
        k: over-generation multiplier
        beta: ratio of importance points
        training: flag

    Returns:
        Training: selected points [B, N, 2]
        Inference: (idx, points) tuple
    """
    assert mask.dim() == 4, "Dim must be N(Batch)CHW"
    device = mask.device
    B, C, H, W = mask.shape

    if C == 1:
        if not training:
            H_step, W_step = 1 / H, 1 / W
            N = min(H * W, N)
            prob = torch.sigmoid(mask[:, 0])
            uncertainty_map = -torch.abs(prob - 0.5)
            _, idx = uncertainty_map.view(B, -1).topk(N, dim=1)
            points = torch.zeros(B, N, 2, dtype=torch.float, device=device)
            points[:, :, 0] = W_step / 2.0 + (idx % W).to(torch.float) * W_step
            points[:, :, 1] = H_step / 2.0 + (idx // W).to(torch.float) * H_step
            return idx, points

        over_generation = torch.rand(B, k * N, 2, device=device)
        over_generation_map = point_sample(mask, over_generation, align_corners=False)
        prob = torch.sigmoid(over_generation_map[:, 0])
        uncertainty_map = -torch.abs(prob - 0.5)
        _, idx = uncertainty_map.topk(int(beta * N), -1)
        shift = (k * N) * torch.arange(B, dtype=torch.long, device=device)
        idx += shift[:, None]
        importance = over_generation.view(-1, 2)[idx.view(-1), :].view(B, int(beta * N), 2)
        coverage = torch.rand(B, N - int(beta * N), 2, device=device)
        return torch.cat([importance, coverage], 1).to(device)
    else:
        mask, _ = mask.sort(1, descending=True)
        if not training:
            H_step, W_step = 1 / H, 1 / W
            N = min(H * W, N)
            uncertainty_map = -1 * (mask[:, 0] - mask[:, 1])
            _, idx = uncertainty_map.view(B, -1).topk(N, dim=1)
            points = torch.zeros(B, N, 2, dtype=torch.float, device=device)
            points[:, :, 0] = W_step / 2.0 + (idx % W).to(torch.float) * W_step
            points[:, :, 1] = H_step / 2.0 + (idx // W).to(torch.float) * H_step
            return idx, points

        over_generation = torch.rand(B, k * N, 2, device=device)
        over_generation_map = point_sample(mask, over_generation, align_corners=False)
        uncertainty_map = -1 * (over_generation_map[:, 0] - over_generation_map[:, 1])
        _, idx = uncertainty_map.topk(int(beta * N), -1)
        shift = (k * N) * torch.arange(B, dtype=torch.long, device=device)
        idx += shift[:, None]
        importance = over_generation.view(-1, 2)[idx.view(-1), :].view(B, int(beta * N), 2)
        coverage = torch.rand(B, N - int(beta * N), 2, device=device)
        return torch.cat([importance, coverage], 1).to(device)


@torch.no_grad()
def sampling_points_dino_guided(mask, dino_sim_map, N, k=3, beta=0.75, dino_ratio=0.25):
    """
    DINO-Guided Point Sampling (Module 3).

    Uncertainty-based importance pool + DINO similarity map guided sampling.

    Args:
        mask: [B, 1, H, W] coarse logits
        dino_sim_map: [B, 1, H_d, W_d] DINO similarity map (upsampled to mask size internally)
        N: total points
        k: over-generation multiplier
        beta: ratio for uncertainty-based importance
        dino_ratio: ratio of DINO-guided points (from the remaining coverage pool)

    Returns:
        points: [B, N, 2] sampled point coordinates in [0,1]
    """
    assert mask.dim() == 4
    device = mask.device
    B, C, H, W = mask.shape

    # Resize dino_sim_map to match mask spatial size if needed
    if dino_sim_map.shape[-2:] != mask.shape[-2:]:
        dino_sim_map = F.interpolate(
            dino_sim_map, size=(H, W), mode='bilinear', align_corners=False
        )

    # 1. Uncertainty-based importance points (same as standard)
    n_importance = int(beta * N)
    n_remaining = N - n_importance
    n_dino = int(dino_ratio * n_remaining)
    n_random = n_remaining - n_dino

    over_generation = torch.rand(B, k * N, 2, device=device)
    over_generation_map = point_sample(mask, over_generation, align_corners=False)
    prob = torch.sigmoid(over_generation_map[:, 0])
    uncertainty_map = -torch.abs(prob - 0.5)
    _, idx = uncertainty_map.topk(n_importance, -1)
    shift = (k * N) * torch.arange(B, dtype=torch.long, device=device)
    idx += shift[:, None]
    importance = over_generation.view(-1, 2)[idx.view(-1), :].view(B, n_importance, 2)

    # 2. DINO-guided points: sample from high-similarity regions
    if n_dino > 0:
        dino_over = torch.rand(B, k * n_dino, 2, device=device)
        dino_values = point_sample(dino_sim_map, dino_over, align_corners=False)  # [B, 1, k*n_dino]
        _, dino_idx = dino_values[:, 0].topk(n_dino, -1)
        dino_shift = (k * n_dino) * torch.arange(B, dtype=torch.long, device=device)
        dino_idx += dino_shift[:, None]
        dino_points = dino_over.view(-1, 2)[dino_idx.view(-1), :].view(B, n_dino, 2)
    else:
        dino_points = torch.zeros(B, 0, 2, device=device)

    # 3. Random coverage
    coverage = torch.rand(B, n_random, 2, device=device)

    return torch.cat([importance, dino_points, coverage], dim=1).to(device)


# ============================================================================
# Module 1: LearnableQueryAdapter (Frozen CrossAttentionQueryExtractor)
# ============================================================================

class LearnableQueryAdapter(nn.Module):
    """
    DINO Feature Adapter based on frozen CrossAttentionQueryExtractor.

    Workflow:
      1. Frozen query module -> dynamic_prototype [B, D], attn_weights [B, 1, H, W]
      2. cosine similarity between dynamic_prototype and DINO features -> sim_map [B, 1, H, W]
      3. MOL-dependent temperature scaling (tau=0.05 for MOL=0, tau=0.1 for MOL=1)
      4. Softmax -> attention weights
      5. Adapter CNN refines -> sigmoid mask
      6. feature_proj(dino * mask) -> injected features

    * Query module is frozen -> no alignment_loss
    """

    def __init__(
        self,
        feature_dim: int = 768,
        hidden_dim: int = 256,
        learnable_query_ckpt: str = "",
        query_embed_dim: int = 768,
        query_num_heads: int = 8,
        query_ffn_dim: int = 2048,
    ):
        super().__init__()
        self.feature_dim = feature_dim

        # ── Frozen CrossAttentionQueryExtractor ──
        self.query_module = CrossAttentionQueryExtractor(
            embed_dim=query_embed_dim,
            num_heads=query_num_heads,
            ffn_dim=query_ffn_dim,
        )
        if learnable_query_ckpt:
            self._load_query_checkpoint(learnable_query_ckpt)
        # Freeze query module
        for p in self.query_module.parameters():
            p.requires_grad = False
        self.query_module.eval()

        # Temperature per MOL class
        self.register_buffer('tau_values', torch.tensor([0.05, 0.1]))

        # Adapter: attention → refined mask
        self.adapter = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1, bias=True),
        )

        # Feature projection after attention
        self.feature_proj = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)

    def _load_query_checkpoint(self, ckpt_path: str):
        """Load CrossAttentionQueryExtractor checkpoint."""
        print(f"Loading CrossAttentionQueryExtractor from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")

        # Extract hyperparameters from saved args (if present)
        if 'args' in ckpt:
            saved_args = ckpt['args']
            if isinstance(saved_args, dict):
                embed_dim = saved_args.get('embed_dim', self.query_module.embed_dim)
                num_heads = saved_args.get('num_heads', 8)
                ffn_dim = saved_args.get('ffn_dim', 2048)
                print(f"  Saved args: embed_dim={embed_dim}, num_heads={num_heads}, ffn_dim={ffn_dim}")

        self.query_module.load_state_dict(ckpt['model_state_dict'])

        if 'epoch' in ckpt:
            print(f"  ✓ Loaded from epoch {ckpt['epoch']}, val_loss={ckpt.get('val_loss', 'N/A')}")
        else:
            print(f"  ✓ Loaded successfully")

    def _get_temperature(self, mol_label: torch.Tensor) -> torch.Tensor:
        """Get per-sample temperature. Returns [B, 1, 1, 1] for broadcasting."""
        tau = self.tau_values[mol_label]  # [B]
        return tau.view(-1, 1, 1, 1)

    def forward(
        self,
        dino_features: torch.Tensor,
        mol_label: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            dino_features: [B, C, H, W] DINO patch features (native resolution)
            mol_label: [B] LongTensor, 0 or 1

        Returns:
            injected_features: [B, C, H, W] (native resolution, resized by caller)
            similarity_map: [B, 1, H, W] (for DINO-guided sampling)
            attn_weights: [B, 1, H, W] (cross-attention map for visualization)
        """
        B, C, H, W = dino_features.shape

        # 1. Frozen query module → dynamic prototype + attention weights
        with torch.no_grad():
            dynamic_prototype, attn_weights = self.query_module(dino_features)
            # dynamic_prototype: [B, D], attn_weights: [B, 1, H, W]

        # 2. Cosine similarity: dynamic_prototype vs all spatial positions
        p_norm = F.normalize(dynamic_prototype, p=2, dim=1).view(B, C, 1, 1)
        f_norm = F.normalize(dino_features, p=2, dim=1)
        similarity_map = (f_norm * p_norm).sum(dim=1, keepdim=True)  # [B, 1, H, W]

        # 3. Temperature-scaled softmax attention
        tau = self._get_temperature(mol_label)
        sim_scaled = similarity_map / tau
        attention_weights = F.softmax(sim_scaled.view(B, 1, -1), dim=-1).view(B, 1, H, W)

        # 4. Adapter CNN refines attention
        attention_mask = self.adapter(attention_weights)
        attention_mask = torch.sigmoid(attention_mask)

        # 5. Feature injection
        injected_features = self.feature_proj(dino_features * attention_mask)

        return injected_features, similarity_map, attn_weights

    def train(self, mode=True):
        """Override train to keep query_module always in eval mode."""
        super().train(mode)
        self.query_module.eval()
        return self


# ============================================================================
# UNet Building Blocks
# ============================================================================

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# ============================================================================
# Module 2: MOL-FiLM PointHead
# ============================================================================

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: F' = γ(mol)·F + β(mol)"""

    def __init__(self, feature_dim: int, mol_embed_dim: int = 32):
        super().__init__()
        self.gamma_proj = nn.Linear(mol_embed_dim, feature_dim)
        self.beta_proj = nn.Linear(mol_embed_dim, feature_dim)

        # Initialize γ→1, β→0 (identity modulation at start)
        nn.init.ones_(self.gamma_proj.weight.data[:, 0])
        nn.init.zeros_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, features: torch.Tensor, mol_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, C, N] (Conv1d feature)
            mol_embed: [B, mol_embed_dim]

        Returns:
            modulated: [B, C, N]
        """
        gamma = self.gamma_proj(mol_embed).unsqueeze(-1)  # [B, C, 1]
        beta = self.beta_proj(mol_embed).unsqueeze(-1)    # [B, C, 1]
        return gamma * features + beta


class MOLFiLMPointHead(nn.Module):
    """
    PointRend Head with FiLM modulation conditioned on MOL.

    Insert FiLM layers after each hidden MLP layer
    to adaptively refine boundaries depending on the MOL class.
    """

    def __init__(
        self,
        num_classes=1,
        in_c=None,
        coarse_channels=256,
        fine_channels=64,
        hidden_dim=256,
        mol_embed_dim=32,
        k=3,
        beta=0.75,
        train_num_points=1024,
        inference_num_points=2048,
    ):
        super().__init__()

        if in_c is None:
            in_c = coarse_channels + fine_channels

        # MOL embedding for FiLM
        self.mol_embedding = nn.Embedding(2, mol_embed_dim)

        # MLP layers (split for FiLM insertion)
        self.conv1 = nn.Conv1d(in_c, hidden_dim, 1)
        self.film1 = FiLMLayer(hidden_dim, mol_embed_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.film2 = FiLMLayer(hidden_dim, mol_embed_dim)
        self.conv_out = nn.Conv1d(hidden_dim, num_classes, 1)

        self.k = k
        self.beta = beta
        self.train_num_points = train_num_points
        self.inference_num_points = inference_num_points

    def _mlp_forward(self, feature_representation, mol_embed):
        """MLP with FiLM modulation."""
        x = self.conv1(feature_representation)
        x = F.relu(x, inplace=True)
        x = self.film1(x, mol_embed)

        x = self.conv2(x)
        x = F.relu(x, inplace=True)
        x = self.film2(x, mol_embed)

        x = self.conv_out(x)
        return x

    def forward(self, fine_grained_features, coarse_logits, mol_label,
                dino_sim_map=None):
        """
        Args:
            fine_grained_features: [B, C_fine, H, W] (512x512)
            coarse_logits: [B, num_classes, H/2, W/2] (256x256)
            mol_label: [B] LongTensor
            dino_sim_map: [B, 1, H_d, W_d] (optional, for guided sampling)

        Returns:
            Training: {"rend": [B, C, N], "points": [B, N, 2]}
            Inference: {"fine": [B, C, H, W]}
        """
        if not self.training:
            return self.inference(fine_grained_features, coarse_logits, mol_label)

        mol_embed = self.mol_embedding(mol_label)  # [B, mol_embed_dim]

        # Training: sample points
        if dino_sim_map is not None:
            points = sampling_points_dino_guided(
                coarse_logits, dino_sim_map,
                N=self.train_num_points,
                k=self.k, beta=self.beta, dino_ratio=0.25,
            )
        else:
            points = sampling_points(
                coarse_logits,
                N=self.train_num_points,
                k=self.k, beta=self.beta, training=True,
            )

        # Sample features at points
        coarse = point_sample(coarse_logits, points, align_corners=False)
        fine = point_sample(fine_grained_features, points, align_corners=False)
        feature_representation = torch.cat([coarse, fine], dim=1)

        # FiLM-modulated MLP
        rend = self._mlp_forward(feature_representation, mol_embed)

        return {"rend": rend, "points": points}

    @torch.no_grad()
    def inference(self, fine_grained_features, coarse_logits, mol_label):
        """
        Inference: subdivision refinement (256→512) with FiLM modulation.
        """
        mol_embed = self.mol_embedding(mol_label)
        out = coarse_logits
        target_h, target_w = fine_grained_features.shape[-2:]

        while out.shape[-1] != target_w or out.shape[-2] != target_h:
            out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=True)

            points_idx, points = sampling_points(
                out, N=self.inference_num_points, training=False
            )

            coarse = point_sample(out, points, align_corners=False)
            fine = point_sample(fine_grained_features, points, align_corners=False)
            feature_representation = torch.cat([coarse, fine], dim=1)

            rend = self._mlp_forward(feature_representation, mol_embed)

            B, C, H, W = out.shape
            points_idx = points_idx.unsqueeze(1).expand(-1, C, -1)
            out = (out.reshape(B, C, -1)
                      .scatter_(2, points_idx, rend)
                      .view(B, C, H, W))

        return {"fine": out}


# ============================================================================
# Main Model: UNet with LearnableQueryAdapter (Frozen CrossAttentionQueryExtractor)
# ============================================================================

class UNetMOLLearnableQuery(nn.Module):
    """
    UNet with MOL-Conditional DINO + PointRend using Pretrained Learnable Query.

    Args:
        n_channels: input channels (1 or 3)
        n_classes: output classes (default 1, binary)
        bilinear: use bilinear upsampling
        use_pointrend: whether to use PointRend
        train_num_points, inference_num_points: PointRend sampling params
        dinov3_model: frozen MedDINOv3 model (optional)
        dinov3_feature_dim: DINO feature dim (default 768)
        dinov3_input_size: DINO input size (default 518)
        learnable_query_ckpt: path to pretrained CrossAttentionQueryExtractor checkpoint
    """

    def __init__(
        self,
        n_channels=1,
        n_classes=1,
        bilinear=False,
        use_pointrend=True,
        train_num_points=1024,
        inference_num_points=2048,
        dinov3_model=None,
        dinov3_feature_dim=768,
        dinov3_input_size=518,
        main_slice_weight=1.0,
        learnable_query_ckpt="",
    ):
        super(UNetMOLLearnableQuery, self).__init__()
        self.n_channels = 3
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_pointrend = use_pointrend
        self.dinov3_model = dinov3_model
        self.dinov3_input_size = dinov3_input_size
        self.dinov3_feature_dim = dinov3_feature_dim
        self.main_slice_weight = main_slice_weight
        factor = 2 if bilinear else 1
        bottleneck_channels = 1024 // factor

        # ── Encoder ──
        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, bottleneck_channels)

        # ── Module 1: LearnableQueryAdapter (Frozen CrossAttentionQueryExtractor) ──
        if dinov3_model is not None:
            self.adapter = LearnableQueryAdapter(
                feature_dim=dinov3_feature_dim,
                hidden_dim=256,
                learnable_query_ckpt=learnable_query_ckpt,
            )
            self.dino_proj = nn.Sequential(
                nn.Conv2d(dinov3_feature_dim, bottleneck_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(bottleneck_channels),
                nn.ReLU(inplace=True),
            )
            self.current_gt_mask = None
            self.current_alignment_loss = None
            self.current_dino_sim_map = None
            self.current_attn_weights = None
        else:
            self.adapter = None
            self.dino_proj = None
            self.current_gt_mask = None
            self.current_alignment_loss = None
            self.current_dino_sim_map = None
            self.current_attn_weights = None

        # ── Decoder ──
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        # Final output head (for full decoder path)
        self.outc = OutConv(64, n_classes)

        # ── PointRend (MOL-FiLM) ──
        self.enable_point_rend = True
        if self.use_pointrend:
            coarse_channels = 128 // factor
            self.coarse_head = OutConv(coarse_channels, n_classes)

            fine_channels = 64
            self.point_head = MOLFiLMPointHead(
                num_classes=n_classes,
                coarse_channels=n_classes,
                fine_channels=fine_channels,
                hidden_dim=256,
                mol_embed_dim=32,
                train_num_points=train_num_points,
                inference_num_points=inference_num_points,
            )

    def _extract_dino_feature(self, slice_data: torch.Tensor, B: int):
        """Extract DINO patch features from a single slice."""
        if self.dinov3_model is None:
            return None
        dinov3_input = F.interpolate(
            slice_data.repeat(1, 3, 1, 1),
            size=(self.dinov3_input_size, self.dinov3_input_size),
            mode='bilinear', align_corners=False,
        )
        with torch.no_grad():
            outputs = self.dinov3_model(dinov3_input, is_training=True)
            patch_tokens = outputs['x_norm_patchtokens']  # [B, N, D]
            N = patch_tokens.shape[1]
            grid_size = int(np.sqrt(N))
            features = patch_tokens.reshape(B, grid_size, grid_size, -1).permute(0, 3, 1, 2)
        return features

    def forward(self, x, mol_label=None, dino_slices=None):
        """
        Args:
            x: [B, 1, H, W] or [B, 3, H, W]
            mol_label: [B] LongTensor (0=normal, 1=MOL). If None, defaults to 0.
            dino_slices: [B, 3, H, W] adjacent slices for DINO (optional).

        Returns:
            Without PointRend:
                logits [B, n_classes, H, W]

            With PointRend (Training, after warmup):
                {
                    "logits": [B, C, H, W],
                    "coarse_logits": [B, C, H/2, W/2],
                    "rend": [B, C, N],
                    "points": [B, N, 2],
                }

            With PointRend (Inference):
                {
                    "logits": [B, C, H, W],
                    "coarse_logits": [B, C, H/2, W/2],
                }
        """
        if x.dim() != 4:
            raise ValueError(f"UNetMOLLearnableQuery expects 4D tensor [B, C, H, W], got shape {x.shape}")

        b, c, h, w = x.shape
        if c == 1:
            x = x.repeat(1, 3, 1, 1)
        elif c != 3:
            raise ValueError(f"UNetMOLLearnableQuery expects 1 or 3 channels, got {c}")

        B = x.shape[0]
        device = x.device

        # Default MOL label: all 0 (normal)
        if mol_label is None:
            mol_label = torch.zeros(B, dtype=torch.long, device=device)

        # ── Encoder ──
        x1 = self.inc(x)       # [B, 64, H, W]       - Fine-grained (512x512)
        x2 = self.down1(x1)    # [B, 128, H/2, W/2]
        x3 = self.down2(x2)    # [B, 256, H/4, W/4]
        x4 = self.down3(x3)    # [B, 512, H/8, W/8]
        x5 = self.down4(x4)    # [B, 1024, H/16, W/16] - bottleneck

        # ── DINO feature injection (MOL-conditioned, frozen learnable query) ──
        dino_sim_map = None
        if self.adapter is not None and self.dinov3_model is not None:
            bottleneck = x5
            _, _, H_bottle, W_bottle = bottleneck.shape
            # Use dino_slices (adjacent slices) if provided, else fallback to x
            dino_source = dino_slices if dino_slices is not None else x
            dino_feat_list = []
            for ch_idx in range(dino_source.shape[1]):
                slice_ch = dino_source[:, ch_idx:ch_idx+1, :, :]
                feat_ch = self._extract_dino_feature(slice_ch, B)
                if feat_ch is not None:
                    dino_feat_list.append(feat_ch)
            if len(dino_feat_list) > 0:
                if self.main_slice_weight != 1.0 and len(dino_feat_list) == 3:
                    w = self.main_slice_weight
                    dino_feat = (dino_feat_list[0] + w * dino_feat_list[1] + dino_feat_list[2]) / (w + 2)
                else:
                    dino_feat = torch.stack(dino_feat_list, dim=0).mean(dim=0)
            else:
                dino_feat = None
            if dino_feat is not None:
                # LearnableQueryAdapter: frozen query → adapter → inject
                injected, sim_map, attn_w = self.adapter(
                    dino_feat, mol_label
                )

                # No alignment_loss (query module is pre-trained)
                self.current_alignment_loss = torch.tensor(0.0, device=device)
                dino_sim_map = sim_map
                self.current_dino_sim_map = sim_map
                self.current_attn_weights = attn_w

                # Resize injected features to bottleneck size
                injected = F.interpolate(
                    injected, size=(H_bottle, W_bottle),
                    mode='bilinear', align_corners=False,
                )
                injected = self.dino_proj(injected)
                x5 = bottleneck + injected

        # ── Decoder ──
        up1_out = self.up1(x5, x4)    # [B, 512, H/8, W/8]
        up2_out = self.up2(up1_out, x3)  # [B, 256, H/4, W/4]
        up3_out = self.up3(up2_out, x2)  # [B, 128, H/2, W/2] (256x256)

        if not self.use_pointrend:
            up4_out = self.up4(up3_out, x1)
            logits = self.outc(up4_out)
            return logits

        # ── PointRend path ──

        # Warmup (training, PointRend disabled): full decoder
        if self.training and not getattr(self, "enable_point_rend", True):
            up4_out = self.up4(up3_out, x1)
            logits = self.outc(up4_out)
            # Also compute coarse for auxiliary loss
            coarse_logits = self.coarse_head(up3_out)
            return {
                "logits": logits,
                "coarse_logits": coarse_logits,
            }

        # Coarse mask from up3 (256x256)
        coarse_logits = self.coarse_head(up3_out)

        # Warmup eval: don't use un-trained point head
        if not self.training and not getattr(self, "enable_point_rend", True):
            up4_out = self.up4(up3_out, x1)
            logits = self.outc(up4_out)
            return {"logits": logits, "coarse_logits": coarse_logits}

        # Fine-grained features (512x512)
        fine_grained_features = x1

        # MOL-FiLM PointHead forward
        point_result = self.point_head(
            fine_grained_features, coarse_logits, mol_label,
            dino_sim_map=dino_sim_map,
        )

        if self.training:
            _, _, H_full, W_full = fine_grained_features.shape
            logits_upsampled = F.interpolate(
                coarse_logits, size=(H_full, W_full),
                mode='bilinear', align_corners=False,
            )
            return {
                "logits": logits_upsampled,
                "coarse_logits": coarse_logits,
                "rend": point_result["rend"],
                "points": point_result["points"],
            }
        else:
            return {
                "logits": point_result["fine"],
                "coarse_logits": coarse_logits,
            }


# ============================================================================
# Build Functions
# ============================================================================

def build_unet_all_mol_learnable_query(
    n_channels=1,
    n_classes=1,
    bilinear=False,
    train_num_points=1024,
    inference_num_points=2048,
    dinov3_model=None,
    dinov3_feature_dim=768,
    dinov3_input_size=518,
    main_slice_weight=1.0,
    learnable_query_ckpt="",
):
    """Build UNet + frozen CrossAttentionQueryExtractor + MOL PointRend (unet_all_mol_learnable_query)."""
    model = UNetMOLLearnableQuery(
        n_channels=n_channels,
        n_classes=n_classes,
        bilinear=bilinear,
        use_pointrend=True,
        train_num_points=train_num_points,
        inference_num_points=inference_num_points,
        dinov3_model=dinov3_model,
        dinov3_feature_dim=dinov3_feature_dim,
        dinov3_input_size=dinov3_input_size,
        main_slice_weight=main_slice_weight,
        learnable_query_ckpt=learnable_query_ckpt,
    )
    return model


# ============================================================================
# Loss Utilities
# ============================================================================

def mol_pointrend_loss_lq(
    output,
    target,
    mol_label,
    bce_weight=0.5,
    dice_weight=0.5,
    coarse_weight=0.5,
    point_weight=1.0,
    adapter_loss_weight=0.0,
    alignment_loss=None,
):
    """
    PointRend Loss for unet_all_mol_learnable_query: uniform point loss.

    L_total = L_seg + coarse_weight * L_aux + point_weight * L_point
    (alignment_loss argument exists for compatibility but is ignored as 0)
    """
    # --- L_seg: main segmentation loss (512x512) ---
    logits = output["logits"]
    loss_bce = F.binary_cross_entropy_with_logits(logits, target)
    pred_prob = torch.sigmoid(logits)
    intersection = (pred_prob * target).sum()
    union = pred_prob.sum() + target.sum()
    loss_dice = 1 - (2.0 * intersection + 1e-6) / (union + 1e-6)
    loss_seg = bce_weight * loss_bce + dice_weight * loss_dice

    # --- L_aux: auxiliary loss (256x256 coarse) ---
    coarse_logits = output["coarse_logits"]
    target_coarse = F.interpolate(
        target, size=coarse_logits.shape[-2:], mode="nearest"
    )
    aux_bce = F.binary_cross_entropy_with_logits(coarse_logits, target_coarse)
    aux_prob = torch.sigmoid(coarse_logits)
    aux_inter = (aux_prob * target_coarse).sum()
    aux_union = aux_prob.sum() + target_coarse.sum()
    aux_dice = 1 - (2.0 * aux_inter + 1e-6) / (aux_union + 1e-6)
    loss_aux = bce_weight * aux_bce + dice_weight * aux_dice

    # --- L_point: uniform point-wise loss ---
    if "rend" in output and "points" in output:
        rend = output["rend"]      # [B, 1, N]
        points = output["points"]  # [B, N, 2]

        target_points = point_sample(target, points, align_corners=False)  # [B, 1, N]
        loss_point = F.binary_cross_entropy_with_logits(rend, target_points)
    else:
        loss_point = torch.tensor(0.0, device=logits.device)

    # --- Total (no alignment loss) ---
    total_loss = (
        loss_seg
        + coarse_weight * loss_aux
        + point_weight * loss_point
    )

    loss_dict = {
        "loss_total": total_loss.item(),
        "loss_seg": loss_seg.item(),
        "loss_aux": loss_aux.item(),
        "loss_point": loss_point.item() if isinstance(loss_point, torch.Tensor) else loss_point,
        "loss_align": 0.0,
    }

    return total_loss, loss_dict


def mol_warmup_loss_lq(
    output,
    target,
    mol_label,
    bce_weight=0.5,
    dice_weight=0.5,
    coarse_weight=0.5,
    adapter_loss_weight=0.0,
    alignment_loss=None,
):
    """
    Warmup phase loss for learnable query model: L_seg + L_aux only.
    (alignment_loss argument exists for compatibility but is ignored as 0)

    Args:
        output: Model output (dict with 'logits' and optionally 'coarse_logits')
        target: GT mask [B, 1, H, W]
        mol_label: [B] LongTensor
        others: loss weights

    Returns:
        total_loss, loss_dict
    """
    logits = output["logits"] if isinstance(output, dict) else output
    loss_bce = F.binary_cross_entropy_with_logits(logits, target)
    pred_prob = torch.sigmoid(logits)
    intersection = (pred_prob * target).sum()
    union = pred_prob.sum() + target.sum()
    loss_dice = 1 - (2.0 * intersection + 1e-6) / (union + 1e-6)
    loss_seg = bce_weight * loss_bce + dice_weight * loss_dice

    # Auxiliary loss
    loss_aux = torch.tensor(0.0, device=logits.device)
    if isinstance(output, dict) and "coarse_logits" in output:
        coarse_logits = output["coarse_logits"]
        target_coarse = F.interpolate(
            target, size=coarse_logits.shape[-2:], mode="nearest"
        )
        aux_bce = F.binary_cross_entropy_with_logits(coarse_logits, target_coarse)
        aux_prob = torch.sigmoid(coarse_logits)
        aux_inter = (aux_prob * target_coarse).sum()
        aux_union = aux_prob.sum() + target_coarse.sum()
        aux_dice = 1 - (2.0 * aux_inter + 1e-6) / (aux_union + 1e-6)
        loss_aux = bce_weight * aux_bce + dice_weight * aux_dice

    total_loss = loss_seg + coarse_weight * loss_aux

    loss_dict = {
        "loss_total": total_loss.item(),
        "loss_seg": loss_seg.item(),
        "loss_aux": loss_aux.item(),
        "loss_align": 0.0,
    }

    return total_loss, loss_dict


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Testing UNetMOLLearnableQuery (Frozen CrossAttentionQueryExtractor)")
    print("=" * 80)

    model = build_unet_all_mol_learnable_query(
        n_channels=1,
        n_classes=1,
        bilinear=False,
        train_num_points=1024,
        inference_num_points=2048,
        dinov3_model=None,
        learnable_query_ckpt="",  # No checkpoint for testing
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    mol_label = torch.tensor([0, 1], dtype=torch.long)

    # Warmup
    print("\n--- Warmup ---")
    model.train()
    model.enable_point_rend = False
    x = torch.randn(2, 1, 512, 512)
    gt = torch.randint(0, 2, (2, 1, 512, 512)).float()
    model.current_gt_mask = gt
    out = model(x, mol_label=mol_label)
    print(f"Keys: {list(out.keys())}")
    loss, ld = mol_warmup_loss_lq(out, gt, mol_label)
    print(f"Loss: {loss.item():.4f} | {ld}")

    # PointRend
    print("\n--- PointRend ---")
    model.enable_point_rend = True
    out2 = model(x, mol_label=mol_label)
    print(f"Keys: {list(out2.keys())}")
    loss2, ld2 = mol_pointrend_loss_lq(out2, gt, mol_label)
    print(f"Loss: {loss2.item():.4f} | {ld2}")

    # Inference
    print("\n--- Inference ---")
    model.eval()
    with torch.no_grad():
        out3 = model(torch.randn(2, 1, 512, 512), mol_label=mol_label)
    print(f"Keys: {list(out3.keys())}")

    print("\nAll tests passed!")
