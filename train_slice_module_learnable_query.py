"""
Standalone Training Script for CrossAttentionQueryExtractor Module
===================================================================

Script to train CrossAttentionQueryExtractor standalone.
Extracts DINO features using frozen MedDINOv3,
and trains learnable query attention maps to match GT Masks.

How to train:
cd /path/to/TISC

python train_slice_module_learnable_query.py --fold 0 --epochs 100 --rotate --elastic --zoom
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from datetime import datetime
import pytz

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Dataset ──
from dataset_2d_slice import (
    TMJ_dataset_2D, collate_fn_2d,
    ElasticTransform, RandomRotation, RandomZoom, ComposeTransforms,
)

# ── Module ──
from models.module_learnable_query import (
    CrossAttentionQueryExtractor,
    compute_attention_loss,
)

# ── MedDINOv3 path ──
MEDDINO_PATH = "/path/to/MedDINOv3"
if MEDDINO_PATH not in sys.path:
    sys.path.insert(0, MEDDINO_PATH)
from dinov3.models.vision_transformer import vit_base

# ── Defaults ──
DEFAULT_DINO_CKPT = "/path/to/MedDINOv3/checkpoint/model.pth"
DEFAULT_RESULT_DIR = "./results/module/learnable_query_aug"
DINO_INPUT_SIZE = 518
DINO_FEATURE_DIM = 768


# ============================================================================
# Helpers
# ============================================================================

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class Logger:
    """Simple logger to file and console."""
    def __init__(self, log_file):
        self.log_file = log_file
        self.terminal = sys.stdout
        with open(self.log_file, 'w') as f:
            f.write(f"Training log started at {datetime.now()}\n")
            f.write("=" * 80 + "\n")

    def write(self, message):
        self.terminal.write(message)
        with open(self.log_file, 'a') as f:
            f.write(message)

    def flush(self):
        self.terminal.flush()


# ============================================================================
# MedDINOv3 Loading
# ============================================================================

def load_dino_model(ckpt_path, device):
    """Load frozen MedDINOv3 model."""
    print(f"Loading MedDINOv3 from: {ckpt_path}")
    model = vit_base(
        drop_path_rate=0.2,
        layerscale_init=1.0e-05,
        n_storage_tokens=4,
        qkv_bias=False,
        mask_k_bias=True,
    )
    chkpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    state_dict = chkpt["teacher"]
    state_dict = {
        k.replace("backbone.", ""): v
        for k, v in state_dict.items()
        if "ibot" not in k and "dino_head" not in k
    }
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    # Freeze all parameters
    for p in model.parameters():
        p.requires_grad = False
    print("  ✓ MedDINOv3 loaded and frozen")
    return model


@torch.no_grad()
def extract_dino_features(dino_model, images, device):
    """
    Extract DINO patch features from input images.

    Args:
        dino_model: frozen MedDINOv3 model
        images: [B, C, H, W] – 1ch or 3ch input images
        device: torch device

    Returns:
        features: [B, 768, H_dino, W_dino] DINO patch features in spatial layout
    """
    B = images.shape[0]
    C = images.shape[1]

    # DINO expects 3ch input
    if C == 1:
        dino_input = images.repeat(1, 3, 1, 1)
    elif C == 3:
        dino_input = images
    else:
        dino_input = images[:, :3, :, :]

    # Resize to DINO input size
    dino_input = F.interpolate(
        dino_input, size=(DINO_INPUT_SIZE, DINO_INPUT_SIZE),
        mode='bilinear', align_corners=False,
    )

    outputs = dino_model(dino_input.to(device), is_training=True)
    patch_tokens = outputs['x_norm_patchtokens']  # [B, N, D]
    N = patch_tokens.shape[1]
    grid_size = int(np.sqrt(N))
    features = patch_tokens.reshape(B, grid_size, grid_size, -1).permute(0, 3, 1, 2)
    # [B, D, grid_size, grid_size] e.g. [B, 768, 37, 37]
    return features


# ============================================================================
# Visualization
# ============================================================================

def save_attn_vis(collected_samples, save_dir, epoch):
    """
    2-panel visualization: GT (Left) vs Attention (Right) for each epoch.
    Saves each sample as a separate image.

    Args:
        collected_samples: list of dict, each containing:
            - 'attn': [1, H_attn, W_attn] raw attention
            - 'gt':   [1, H_gt, W_gt] binary GT mask
            - 'img':  [C, H, W] original input image
            - 'pid':  patient_id (str)
            - 'side': 'right' or 'left'
        save_dir: directory to save visualizations
        epoch: current epoch number
    """
    epoch_dir = os.path.join(save_dir, f"epoch_{epoch:03d}")
    os.makedirs(epoch_dir, exist_ok=True)

    for idx, sample in enumerate(collected_samples):
        attn = sample['attn'].cpu()    # [1, H_a, W_a]
        gt = sample['gt'].cpu()        # [1, H_g, W_g]
        img = sample['img'].cpu()      # [C, H, W]
        pid = sample.get('pid', '')
        side = sample.get('side', '')

        H_attn, W_attn = attn.shape[-2:]

        # Min-max normalize attention → [0, 1]
        a_flat = attn.view(-1)
        a_min, a_max = a_flat.min(), a_flat.max()
        attn_norm = (attn - a_min) / (a_max - a_min + 1e-8)

        # Downsample GT to attention resolution
        # gt_down = F.interpolate(
        #     gt.unsqueeze(0).float(), size=(H_attn, W_attn), mode='nearest'
        # )
        gt_down = F.adaptive_max_pool2d(gt.unsqueeze(0).float(), output_size=(H_attn, W_attn)).squeeze(0)  # [1, H_a, W_a]

        # ── 2-panel: GT (left) | Attention (right) ──
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        axes[0].imshow(gt_down[0].numpy(), cmap='gray', vmin=0, vmax=1)
        axes[0].set_title("GT Mask", fontsize=11)
        axes[0].axis('off')

        axes[1].imshow(attn_norm[0].numpy(), cmap='jet', vmin=0, vmax=1)
        axes[1].set_title("Attention Map", fontsize=11)
        axes[1].axis('off')

        title = f"Epoch {epoch}"
        if pid:
            title += f"  |  {pid}"
        if side:
            title += f" ({side})"
        plt.suptitle(title, fontsize=12)
        plt.tight_layout()

        fname = f"sample_{idx:02d}"
        if pid:
            fname += f"_{pid}"
        if side:
            fname += f"_{side}"
        fname += ".png"
        plt.savefig(os.path.join(epoch_dir, fname), dpi=100, bbox_inches='tight')
        plt.close()


# ============================================================================
# Training & Validation Loops
# ============================================================================

def train_one_epoch(query_module, dino_model, train_loader, optimizer, device, epoch, args):
    """Train one epoch."""
    query_module.train()

    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}/{args.epochs}")
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)  # [B, 1, H, W]

        # 1. Extract DINO features (no grad)
        dino_features = extract_dino_features(dino_model, images, device)

        # 2. Forward through query module
        dynamic_prototype, attn_weights = query_module(dino_features)

        # 3. Attention supervision loss
        loss = compute_attention_loss(
            attn_weights, labels,
            bce_weight=args.bce_weight,
            dice_weight=args.dice_weight,
        )

        # 4. Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(query_module.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


@torch.no_grad()
def validate_one_epoch(query_module, dino_model, val_loader, device, epoch, args):
    """
    Validate one epoch and collect visualization samples.
    Saves up to save_attn_max samples at intervals of save_attn_epoch.
    """
    query_module.eval()

    total_loss = 0.0
    num_batches = 0

    # Determine whether to collect visualization samples
    should_save = (
        args.save_attn_epoch > 0 and epoch % args.save_attn_epoch == 0
    )
    collected = []  # list of sample dicts

    for batch in tqdm(val_loader, desc="Validation"):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        dino_features = extract_dino_features(dino_model, images, device)
        dynamic_prototype, attn_weights = query_module(dino_features)

        loss = compute_attention_loss(
            attn_weights, labels,
            bce_weight=args.bce_weight,
            dice_weight=args.dice_weight,
        )

        total_loss += loss.item()
        num_batches += 1

        # -- Collect visualization samples (only those with non-empty GT) --
        if should_save and len(collected) < args.save_attn_max:
            B = images.shape[0]
            for i in range(B):
                if len(collected) >= args.save_attn_max:
                    break
                # Collect only non-empty GT samples
                if labels[i].sum() > 0:
                    collected.append({
                        'attn': attn_weights[i].detach(),  # [1, H, W]
                        'gt':   labels[i].detach(),        # [1, H, W]
                        'img':  images[i].detach(),        # [C, H, W]
                        'pid':  batch['patient_id'][i] if 'patient_id' in batch else '',
                        'side': batch['side'][i] if 'side' in batch else '',
                    })

    avg_loss = total_loss / max(num_batches, 1)

    # -- Save visualizations --
    if should_save and len(collected) > 0:
        vis_dir = os.path.join(args.result_dir, f"fold{args.fold}", "vis")
        save_attn_vis(collected, vis_dir, epoch)
        print(f"  📷 Saved {len(collected)} attention visualizations → {vis_dir}/epoch_{epoch:03d}/")

    return avg_loss


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Standalone training for CrossAttentionQueryExtractor module"
    )

    # ── Data ──
    parser.add_argument("--fold", type=int, default=0, help="Fold index (0-4)")
    parser.add_argument("--target_size", type=int, default=512, help="Image resize target")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")

    # ── Training ──
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--bce_weight", type=float, default=0.5, help="BCE loss weight")
    parser.add_argument("--dice_weight", type=float, default=0.0, help="Dice loss weight")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume training")

    # ── Model ──
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--ffn_dim", type=int, default=2048)

    # ── DINO ──
    parser.add_argument("--dino_ckpt", type=str, default=DEFAULT_DINO_CKPT)

    # ── Output ──
    parser.add_argument("--result_dir", type=str, default=DEFAULT_RESULT_DIR,
                        help="Result directory")
    parser.add_argument("--save_attn_epoch", type=int, default=1,
                        help="Interval in epochs to save attention visualization (0 to disable).")
    parser.add_argument("--save_attn_max", type=int, default=5,
                        help="Max number of samples to visualize per epoch.")
    parser.add_argument("--save_epochly", action="store_true", default=False,
                        help="Save checkpoint each epoch.")

    # ── Augmentation ──
    parser.add_argument("--augment", action="store_true", default=True,
                        help="Use data augmentation (legacy, applies all default augmentations)")
    
    parser.add_argument('--rotate', action='store_true',
                        help='Apply random rotation augmentation during training (-15 to +15 degrees, p=0.5)')
    parser.add_argument('--rotate_degrees', type=float, default=15.0,
                        help='Rotation angle range +/- degrees (default: 15.0)')
    parser.add_argument('--rotate_p', type=float, default=0.5,
                        help='Rotation probability. Default: 0.5')
    
    parser.add_argument('--elastic', action='store_true',
                        help='Apply elastic deformation augmentation during training (alpha=80, sigma=8, p=0.5)')
    parser.add_argument('--elastic_alpha', type=float, default=80.0,
                        help='Elastic deformation alpha (strength). Default: 80.0')
    parser.add_argument('--elastic_sigma', type=float, default=8.0,
                        help='Elastic deformation sigma (smoothness). Default: 8.0')
    parser.add_argument('--elastic_p', type=float, default=0.5,
                        help='Elastic deformation probability. Default: 0.5')
                        
    parser.add_argument('--zoom', action='store_true',
                        help='Apply random zoom augmentation during training (center crop after scale)')
    parser.add_argument('--zoom_min', type=float, default=0.0,
                        help='Zoom minimum scale fraction (0.0 = 1.0x). Default: 0.0')
    parser.add_argument('--zoom_max', type=float, default=0.1,
                        help='Zoom maximum scale fraction (0.1 = 1.1x). Default: 0.1')
    parser.add_argument('--zoom_p', type=float, default=0.5,
                        help='Zoom probability. Default: 0.5')

    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Result directory ──
    fold_dir = os.path.join(args.result_dir, f"fold{args.fold}")
    ckpt_dir = os.path.join(fold_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Logger
    logger = Logger(os.path.join(fold_dir, "training_log.txt"))
    logger.write(f"\n{'='*80}\n")
    logger.write(f"CrossAttentionQueryExtractor Training\n")
    logger.write(f"Fold: {args.fold}\n")
    logger.write(f"Epochs: {args.epochs}, LR: {args.lr}\n")
    logger.write(f"BCE weight: {args.bce_weight}, Dice weight: {args.dice_weight}\n")
    logger.write(f"Embed dim: {args.embed_dim}, Heads: {args.num_heads}, FFN: {args.ffn_dim}\n")
    logger.write(f"{'='*80}\n\n")

    # ── Data augmentation ──
    train_transforms = []
    
    if args.augment:
        print("  Legacy --augment enabled, turning on rotate, elastic, zoom by default.")
        args.rotate = True
        args.elastic = True
        args.zoom = True

    if args.elastic:
        train_transforms.append(
            ElasticTransform(alpha=args.elastic_alpha, sigma=args.elastic_sigma, p=args.elastic_p)
        )
        print(f"  Augmentation: ElasticTransform(alpha={args.elastic_alpha}, sigma={args.elastic_sigma}, p={args.elastic_p})")
        
    if args.rotate:
        train_transforms.append(
            RandomRotation(degree_range=(-args.rotate_degrees, args.rotate_degrees), p=args.rotate_p)
        )
        print(f"  Augmentation: RandomRotation(degrees=+/-{args.rotate_degrees}, p={args.rotate_p})")

    if args.zoom:
        train_transforms.append(
            RandomZoom(scale_range=(args.zoom_min, args.zoom_max), p=args.zoom_p)
        )
        print(f"  Augmentation: RandomZoom(range=[{args.zoom_min}, {args.zoom_max}], p={args.zoom_p})")
        
    train_transform = ComposeTransforms(train_transforms) if train_transforms else None

    # ── Datasets ──
    target_size = (args.target_size, args.target_size)

    train_dataset = TMJ_dataset_2D(
        split='train',
        fold=args.fold,
        normalize='minmax',
        target_size=target_size,
        transform=train_transform,
    )
    val_dataset = TMJ_dataset_2D(
        split='val',
        fold=args.fold,
        normalize='minmax',
        target_size=target_size,
        transform=None,
    )

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn_2d,
        worker_init_fn=seed_worker, generator=g, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn_2d,
        pin_memory=True,
    )

    logger.write(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}\n")
    logger.write(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}\n\n")

    # ── Load DINO ──
    dino_model = load_dino_model(args.dino_ckpt, device)

    # ── Create Query Module ──
    query_module = CrossAttentionQueryExtractor(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
    ).to(device)

    num_params = sum(p.numel() for p in query_module.parameters() if p.requires_grad)
    logger.write(f"Query module parameters: {num_params:,}\n\n")

    # ── Optimizer & Scheduler ──
    optimizer = torch.optim.AdamW(
        query_module.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-4,
    )

    # ── Resume Checkpoint ──
    start_epoch = 1
    best_val_loss = float('inf')

    if args.resume and os.path.exists(args.resume):
        try:
            logger.write(f"Resuming from checkpoint: {args.resume}\n")
            ckpt = torch.load(args.resume, map_location=device)
            query_module.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            else:
                # If scheduler state is missing, advance it to the correct epoch
                for _ in range(ckpt['epoch']):
                    scheduler.step()
            start_epoch = ckpt['epoch'] + 1
            best_val_loss = ckpt.get('val_loss', best_val_loss)
            logger.write(f"  ✓ Resumed from epoch {ckpt['epoch']}, best_val_loss={best_val_loss:.6f}\n\n")
        except Exception as e:
            logger.write(f"  ⚠ Failed to resume checkpoint: {e}\n\n")

    # ── Training Loop ──
    for epoch in range(start_epoch, args.epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        logger.write(f"Epoch [{epoch}/{args.epochs}]  lr={current_lr:.6f}\n")

        # Train
        train_loss = train_one_epoch(
            query_module, dino_model, train_loader, optimizer, device, epoch, args
        )

        # Validate
        val_loss = validate_one_epoch(
            query_module, dino_model, val_loader, device, epoch, args
        )

        scheduler.step()

        logger.write(
            f"  Train Loss: {train_loss:.6f}  |  Val Loss: {val_loss:.6f}\n"
        )

        # ── Save best checkpoint ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(ckpt_dir, "best_query_module.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': query_module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'args': vars(args),
            }, best_path)
            logger.write(f"  ★ Best model saved (val_loss={val_loss:.6f})\n")

        # ── Save epoch checkpoint ──
        if args.save_epochly or epoch % 10 == 0:
            epoch_path = os.path.join(ckpt_dir, f"epoch_{epoch:03d}_{val_loss:.4f}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': query_module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'args': vars(args),
            }, epoch_path)

        logger.write("\n")

    logger.write(f"\nTraining complete. Best val loss: {best_val_loss:.6f}\n")
    logger.write(f"Results saved to: {fold_dir}\n")
    print(f"\n✓ Training complete. Results saved to: {fold_dir}")
    print(f"  Best val loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
