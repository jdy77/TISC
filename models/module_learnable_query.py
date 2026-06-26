"""
Cross-Attention Learnable Query Module for Dynamic Prototype Extraction
========================================================================

Learnable Query module based on Transformer Decoder Block.
Takes frozen DINOv3 feature maps to output dynamic prototype vectors
and attention maps focusing on the disc region.

This module can be pre-trained independently and integrated as frozen.

How to train:
  cd /path/to/TISC
  python train_slice_module_learnable_query.py --fold 0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import torchvision.transforms.functional as TF


class CrossAttentionQueryExtractor(nn.Module):
    """
    Cross-Attention based dynamic prototype extractor.

    A single Learnable Query cross-attends DINO features (Key, Value)
    to generate image-adaptive prototype vectors.

    Architecture:
        Q = learnable_query.repeat(B)             → [B, 1, D]
        K = V = dino_features.flatten(spatial)     → [B, H*W, D]
        Cross-Attention → Add & Norm → FFN → Add & Norm
        → dynamic_prototype [B, D]
        → attn_weights [B, 1, H, W]

    Args:
        embed_dim: DINO feature dimension (default 768)
        num_heads: number of attention heads (default 8)
        ffn_dim: FFN hidden dimension (default 2048)
        dropout: dropout rate (default 0.0)
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # ── Learnable Query: [1, 1, D] ──
        self.query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.query, std=0.02)

        # ── Multi-Head Cross-Attention ──
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)

        # ── Feed-Forward Network ──
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self, dino_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            dino_features: [B, D, H, W] — frozen DINO patch features

        Returns:
            dynamic_prototype: [B, D] — image-adaptive prototype vector
            attn_weights:      [B, 1, H, W] — attention score map (2D reshaped)
        """
        B, D, H, W = dino_features.shape

        # Key, Value: [B, H*W, D]
        kv = dino_features.flatten(2).permute(0, 2, 1)  # [B, H*W, D]

        # Query: [1, 1, D] → [B, 1, D]
        q = self.query.expand(B, -1, -1)

        # ── Cross-Attention ──
        attn_out, attn_weights_raw = self.cross_attn(
            query=q,        # [B, 1, D]
            key=kv,          # [B, H*W, D]
            value=kv,        # [B, H*W, D]
            need_weights=True,
            average_attn_weights=True,  # average over heads → [B, 1, H*W]
        )
        # attn_out: [B, 1, D]
        # attn_weights_raw: [B, 1, H*W]

        # Add & Norm
        q = self.norm1(q + attn_out)  # [B, 1, D]

        # ── FFN ──
        ffn_out = self.ffn(q)
        q = self.norm2(q + ffn_out)  # [B, 1, D]

        # # ── Outputs ──
        # dynamic_prototype = q.squeeze(1)  # [B, D]

        # # Attention map: [B, 1, H*W] → [B, 1, H, W]
        # attn_weights = attn_weights_raw.view(B, 1, H, W)

        # return dynamic_prototype, attn_weights
        
        # ── Outputs ──
        # Attention map: [B, 1, H*W] → [B, 1, H, W]
        attn_weights = attn_weights_raw.view(B, 1, H, W)
        
        # * Modified *
        # Instead of q, use attn_weights to take a weighted sum
        # of original DINO features over the disc region.
        dynamic_prototype = (dino_features * attn_weights).sum(dim=(2, 3))  # [B, D]

        return dynamic_prototype, attn_weights


# def compute_attention_loss(
#     attn_weights: torch.Tensor,
#     gt_mask: torch.Tensor,
#     bce_weight: float = 0.5,
#     dice_weight: float = 0.5,
#     smooth: float = 1e-6,
# ) -> torch.Tensor:
#     """
#     Attention Supervision Loss: BCE + Dice between normalized attention map and GT mask.
#
#     Automatically skip empty masks.
#
#     Args:
#         attn_weights: [B, 1, H_attn, W_attn] — raw attention from MHCA (sum=1 distribution)
#         gt_mask:      [B, 1, H_gt, W_gt]     — binary GT mask (0 or 1)
#         bce_weight:   BCE loss weight
#         dice_weight:  Dice loss weight
#         smooth:       Dice smoothing factor
#
#     Returns:
#         loss: scalar tensor (BCE + Dice)
#     """
#     B = attn_weights.shape[0]
#     H_attn, W_attn = attn_weights.shape[-2:]
#
#     # Downsample GT to attention resolution
#     gt_down = F.interpolate(
#         gt_mask.float(), size=(H_attn, W_attn), mode='nearest'
#     )  # [B, 1, H_attn, W_attn]
#
#     # Min-max normalize attention map
#     attn_flat = attn_weights.view(B, -1)  # [B, H*W]
#     attn_min = attn_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
#     attn_max = attn_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
#     attn_norm = (attn_weights - attn_min) / (attn_max - attn_min + 1e-8)
#     # attn_norm: [B, 1, H_attn, W_attn], value range [0, 1]
#
#     # -- Filter empty masks --
#     # Exclude empty GT samples from loss
#     gt_sums = gt_down.view(B, -1).sum(dim=1)  # [B]
#     valid_mask = gt_sums > 0  # [B] bool
#     n_valid = valid_mask.sum().item()
#
#     if n_valid == 0:
#         return torch.tensor(0.0, device=attn_weights.device, requires_grad=True)
#
#     attn_valid = attn_norm[valid_mask]  # [N_valid, 1, H, W]
#     gt_valid = gt_down[valid_mask]      # [N_valid, 1, H, W]
#
#     # ── BCE Loss ──
#     # BCE with normalized attention
#     loss_bce = F.binary_cross_entropy(
#         attn_valid.clamp(1e-7, 1 - 1e-7), gt_valid, reduction='mean'
#     )
#
#     # ── Dice Loss ──
#     intersection = (attn_valid * gt_valid).sum()
#     union = attn_valid.sum() + gt_valid.sum()
#     loss_dice = 1.0 - (2.0 * intersection + smooth) / (union + smooth)
#
#     loss = bce_weight * loss_bce + dice_weight * loss_dice
#     return loss



# ===========================================================

def compute_attention_loss(
    attn_weights: torch.Tensor,
    gt_mask: torch.Tensor,
    bce_weight: float = 1.0,  # High BCE weight is recommended for soft targets
    dice_weight: float = 0.5, # (Optional) Soft Dice
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    Attention Supervision Loss with Gaussian Soft Target.
    Compares attention map with a soft target (1.0 inside mask, Gaussian decay outside).
    """
    B = attn_weights.shape[0]
    H_attn, W_attn = attn_weights.shape[-2:]

    # 1. Downsample GT to attention resolution (32x32)
    # gt_down = F.interpolate(
    #     gt_mask.float(), size=(H_attn, W_attn), mode='nearest'
    # )
    gt_down = F.adaptive_max_pool2d(
        gt_mask.float(), output_size=(H_attn, W_attn)
    )

    # 2. Filter empty masks
    gt_sums = gt_down.view(B, -1).sum(dim=1)
    valid_mask = gt_sums > 0
    n_valid = valid_mask.sum().item()

    if n_valid == 0:
        return torch.tensor(0.0, device=attn_weights.device, requires_grad=True)

    attn_valid = attn_weights[valid_mask]
    gt_valid = gt_down[valid_mask]

    # ====================================================================
    # 3. Create Gaussian-blurred Soft Target
    # ====================================================================
    # Control decay range with kernel size and sigma (e.g. 7x7, sigma=2.0)
    blurred_gt = TF.gaussian_blur(gt_valid, kernel_size=[7, 7], sigma=[2.0, 2.0])

    # Normalize max pixel value to 1.0
    b_max = blurred_gt.view(n_valid, -1).max(dim=1)[0].view(-1, 1, 1, 1) + 1e-8
    blurred_gt_norm = blurred_gt / b_max

    # Keep inside mask as 1.0, apply decay to outer region
    soft_gt = torch.max(gt_valid, blurred_gt_norm)
    # ====================================================================

    # 4. Min-max normalize attention map to [0, 1]
    attn_flat = attn_valid.view(n_valid, -1)
    attn_min = attn_flat.min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
    attn_max = attn_flat.max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
    attn_norm = (attn_valid - attn_min) / (attn_max - attn_min + 1e-8)

    # 5. Calculate Loss
    # BCE is highly effective for soft targets
    loss_bce = F.binary_cross_entropy(
        attn_norm.clamp(1e-7, 1 - 1e-7), soft_gt, reduction='mean'
    )

    # Soft Dice (Optional)
    intersection = (attn_norm * soft_gt).sum()
    union = attn_norm.sum() + soft_gt.sum()
    loss_dice = 1.0 - (2.0 * intersection + smooth) / (union + smooth)

    loss = bce_weight * loss_bce + dice_weight * loss_dice
    return loss

