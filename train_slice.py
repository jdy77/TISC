"""
UNet Training Script for TMJ 2D Segmentation
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
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from scipy.ndimage import distance_transform_edt

from dataset_2d_slice import TMJ_dataset_2D, collate_fn_2d, ElasticTransform, RandomRotation, RandomZoom, ComposeTransforms
from models import get_model
from models.model_unet_all_mol_learnable_query import mol_pointrend_loss_lq, mol_warmup_loss_lq



def seed_everything(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def seed_worker(worker_id):
    """Seed worker for DataLoader"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def dice_coefficient(pred, target, smooth=1e-6):
    """
    Calculate Dice coefficient
    
    Args:
        pred: Predicted mask [B, 1, H, W]
        target: Ground truth mask [B, 1, H, W]
        smooth: Smoothing factor
    
    Returns:
        Dice coefficient (scalar)
    """
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice


def dice_loss(pred, target, smooth=1e-6):
    """
    Dice loss for binary segmentation
    
    Args:
        pred: Predicted logits [B, 1, H, W]
        target: Ground truth mask [B, 1, H, W]
        smooth: Smoothing factor
    
    Returns:
        Dice loss (scalar)
    """
    pred = torch.sigmoid(pred)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice


def combined_loss(pred, target, bce_weight=0.5, dice_weight=0.5):
    """
    Combined BCE + Dice loss
    
    Args:
        pred: Predicted logits [B, 1, H, W]
        target: Ground truth mask [B, 1, H, W]
        bce_weight: Weight for BCE loss
        dice_weight: Weight for Dice loss
    
    Returns:
        Combined loss
    """
    bce = F.binary_cross_entropy_with_logits(pred, target)
    dice = dice_loss(pred, target)
    
    return bce_weight * bce + dice_weight * dice


# ============================================================================
# clDice Loss with Soft-Skeletonization
# ============================================================================

def soft_erode(img):
    """Differentiable erosion via min-pooling (3x3)."""
    p = F.max_pool2d(-img, kernel_size=3, stride=1, padding=1)
    return -p


def soft_dilate(img):
    """Differentiable dilation via max-pooling (3x3)."""
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)


def soft_open(img):
    """Soft morphological opening = erode then dilate."""
    return soft_dilate(soft_erode(img))


def soft_skeletonize(img, num_iter=10):
    """
    Differentiable skeletonization via iterative thinning.
    
    skeleton = img - open(img) at each scale, accumulated.
    
    Args:
        img: [B, 1, H, W] soft probability mask (0~1)
        num_iter: number of thinning iterations
    
    Returns:
        skeleton: [B, 1, H, W] soft skeleton
    """
    img_clone = img.clone()
    skel = torch.zeros_like(img)
    
    for _ in range(num_iter):
        opened = soft_open(img_clone)
        delta = torch.clamp(img_clone - opened, min=0.0)
        skel = torch.clamp(skel + delta, max=1.0)
        img_clone = soft_erode(img_clone)
    
    return skel



def cldice_coefficient(pred_logits, target, num_iter=10, smooth=1e-6):
    """
    clDice metric (centerline Dice coefficient).
    
    Args:
        pred_logits: [B, 1, H, W] predicted logits
        target: [B, 1, H, W] GT mask
    
    Returns:
        clDice coefficient (scalar, higher is better)
    """
    pred = torch.sigmoid(pred_logits)
    pred_bin = (pred > 0.5).float()
    
    skel_pred = soft_skeletonize(pred_bin, num_iter=num_iter)
    skel_gt = soft_skeletonize(target, num_iter=num_iter)
    
    tprec = (skel_pred * target).sum(dim=(2, 3)) / (skel_pred.sum(dim=(2, 3)) + smooth)
    tsens = (skel_gt * pred_bin).sum(dim=(2, 3)) / (skel_gt.sum(dim=(2, 3)) + smooth)
    
    cldice = 2.0 * (tprec * tsens) / (tprec + tsens + smooth)
    return cldice.mean()


def centroid_distance_loss(pred_logits, target, smooth=1e-6):
    """
    Differentiable centroid distance loss.
    
    Computes soft centroid from sigmoid probability map for gradient flow.
    GT centroid is hard centroid of binary mask.
    Loss = L2 distance between predicted and GT centroids / image diagonal.
    
    Args:
        pred_logits: [B, 1, H, W] predicted logits
        target: [B, 1, H, W] GT mask (binary)
    
    Returns:
        Normalized centroid distance loss (scalar, lower is better)
    """
    B, _, H, W = pred_logits.shape
    device = pred_logits.device
    
    # Create coordinate grids [H, W]
    y_coords = torch.arange(H, dtype=torch.float32, device=device).view(H, 1).expand(H, W)
    x_coords = torch.arange(W, dtype=torch.float32, device=device).view(1, W).expand(H, W)
    
    pred_prob = torch.sigmoid(pred_logits)  # [B, 1, H, W], differentiable
    
    total_loss = torch.tensor(0.0, device=device)
    valid_count = 0
    
    for i in range(B):
        p = pred_prob[i, 0]    # [H, W] soft prediction
        g = target[i, 0]       # [H, W] GT binary mask
        
        # Skip if GT is empty
        g_sum = g.sum()
        if g_sum < 1.0:
            continue
        
        p_sum = p.sum() + smooth
        
        # Predicted centroid (soft, differentiable)
        pred_cy = (p * y_coords).sum() / p_sum
        pred_cx = (p * x_coords).sum() / p_sum
        
        # GT centroid (hard)
        gt_cy = (g * y_coords).sum() / g_sum
        gt_cx = (g * x_coords).sum() / g_sum
        
        # L2 distance, normalized by image diagonal
        diag = (H ** 2 + W ** 2) ** 0.5
        dist = ((pred_cy - gt_cy) ** 2 + (pred_cx - gt_cx) ** 2) ** 0.5
        total_loss = total_loss + dist / diag
        valid_count += 1
    
    if valid_count > 0:
        return total_loss / valid_count
    else:
        return torch.tensor(0.0, device=device, requires_grad=True)


def compute_hd95(pred_binary_np, target_np):
    """
    95% Hausdorff Distance (HD95) for a single 2D binary mask pair.
    
    Args:
        pred_binary_np: [H, W] numpy binary prediction
        target_np: [H, W] numpy binary GT
    
    Returns:
        HD95 value (float). Returns np.nan if either mask is empty.
    """
    pred_bool = pred_binary_np.astype(bool)
    target_bool = target_np.astype(bool)
    
    if not pred_bool.any() or not target_bool.any():
        return np.nan
    
    # Surface distances: distance from pred boundary to nearest GT boundary
    # distance_transform_edt: distance of each pixel to nearest foreground pixel
    dt_target = distance_transform_edt(~target_bool)
    dt_pred = distance_transform_edt(~pred_bool)
    
    # Boundary pixels
    from scipy.ndimage import binary_erosion
    pred_boundary = pred_bool ^ binary_erosion(pred_bool)
    target_boundary = target_bool ^ binary_erosion(target_bool)
    
    if not pred_boundary.any() or not target_boundary.any():
        return np.nan
    
    # Surface distances
    dist_pred_to_gt = dt_target[pred_boundary]   # pred boundary → nearest GT
    dist_gt_to_pred = dt_pred[target_boundary]    # GT boundary → nearest pred
    
    all_distances = np.concatenate([dist_pred_to_gt, dist_gt_to_pred])
    return np.percentile(all_distances, 95)


def compute_assd(pred_binary_np, target_np):
    """
    Average Symmetric Surface Distance (ASSD) for a single 2D binary mask pair.
    
    Returns:
        ASSD value (float). Returns np.nan if either mask is empty.
    """
    pred_bool = pred_binary_np.astype(bool)
    target_bool = target_np.astype(bool)
    
    if not pred_bool.any() or not target_bool.any():
        return np.nan
    
    dt_target = distance_transform_edt(~target_bool)
    dt_pred = distance_transform_edt(~pred_bool)
    
    from scipy.ndimage import binary_erosion
    pred_boundary = pred_bool ^ binary_erosion(pred_bool)
    target_boundary = target_bool ^ binary_erosion(target_bool)
    
    if not pred_boundary.any() or not target_boundary.any():
        return np.nan
    
    dist_pred_to_gt = dt_target[pred_boundary]
    dist_gt_to_pred = dt_pred[target_boundary]
    
    assd = (dist_pred_to_gt.mean() + dist_gt_to_pred.mean()) / 2.0
    return assd


def compute_centroid_distance(pred_binary_np, target_np):
    """
    Euclidean distance between centroids of predicted and GT masks.
    
    Returns:
        Centroid distance (float). Returns np.nan if either mask is empty.
    """
    pred_bool = pred_binary_np.astype(bool)
    target_bool = target_np.astype(bool)
    
    if not pred_bool.any() or not target_bool.any():
        return np.nan
    
    # Centroid: mean of (y, x) coordinates of foreground pixels
    pred_coords = np.argwhere(pred_bool)   # [N, 2] (row, col)
    gt_coords = np.argwhere(target_bool)
    
    pred_centroid = pred_coords.mean(axis=0)  # [2]
    gt_centroid = gt_coords.mean(axis=0)
    
    return np.sqrt(((pred_centroid - gt_centroid) ** 2).sum())


def compute_surface_metrics_batch(logits, labels):
    """
    Compute HD95, ASSD, Centroid Distance for a batch.
    
    Args:
        logits: [B, 1, H, W] predicted logits (torch tensor)
        labels: [B, 1, H, W] GT masks (torch tensor)
    
    Returns:
        dict with 'hd95', 'assd', 'centroid_dist' (batch averages, ignoring NaN)
    """
    pred_binary = (torch.sigmoid(logits) > 0.5).float()
    pred_np = pred_binary.cpu().numpy()
    label_np = labels.cpu().numpy()
    
    B = pred_np.shape[0]
    hd95_list, assd_list, cd_list = [], [], []
    
    for i in range(B):
        p = pred_np[i, 0]  # [H, W]
        g = label_np[i, 0]
        hd95_list.append(compute_hd95(p, g))
        assd_list.append(compute_assd(p, g))
        cd_list.append(compute_centroid_distance(p, g))
    
    return {
        'hd95': float(np.nanmean(hd95_list)) if hd95_list else 0.0,
        'assd': float(np.nanmean(assd_list)) if assd_list else 0.0,
        'centroid_dist': float(np.nanmean(cd_list)) if cd_list else 0.0,
    }


class Logger:
    """Simple logger to file and console"""
    
    def __init__(self, log_file):
        self.log_file = log_file
        self.terminal = sys.stdout
        
        # Create log file
        with open(self.log_file, 'w') as f:
            f.write(f"Training log started at {datetime.now()}\n")
            f.write("=" * 80 + "\n")
    
    def write(self, message):
        """Write message to both console and file"""
        self.terminal.write(message)
        with open(self.log_file, 'a') as f:
            f.write(message)
    
    def flush(self):
        """Flush output"""
        self.terminal.flush()


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, args, logger):
    """Train one epoch"""
    model.train()
    
    total_loss = 0.0
    total_dice = 0.0
    total_cldice = 0.0
    num_batches = 0
    
    # --log_loss: Accumulate loss components per epoch
    loss_accum = {}  # key -> float value
    
    if hasattr(model, "enable_point_rend"):
        model.enable_point_rend = (args.warm_up == 0 or epoch > args.warm_up)

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        dino_slices = batch.get('dino_slices')
        if dino_slices is not None:
            dino_slices = dino_slices.to(device)

        if hasattr(model, 'current_gt_mask'):
            model.current_gt_mask = labels

        has_dino = hasattr(model, 'dinov3_model') and model.dinov3_model is not None
        mol_label = batch['mol'].to(device)
        model.current_gt_mask = labels
        if has_dino:
            outputs = model(images, mol_label=mol_label, dino_slices=dino_slices)
        else:
            outputs = model(images, mol_label=mol_label)

        alignment_loss = getattr(model, 'current_alignment_loss', None)
        if isinstance(outputs, dict) and 'rend' in outputs:
            total_loss_value, loss_dict = mol_pointrend_loss_lq(
                outputs, labels, mol_label,
                bce_weight=args.bce_weight,
                dice_weight=args.dice_weight,
                coarse_weight=args.coarse_weight,
                point_weight=args.point_loss_weight,
                adapter_loss_weight=0.0,
                alignment_loss=alignment_loss,
            )
        else:
            total_loss_value, loss_dict = mol_warmup_loss_lq(
                outputs, labels, mol_label,
                bce_weight=args.bce_weight,
                dice_weight=args.dice_weight,
                coarse_weight=args.coarse_weight,
                adapter_loss_weight=0.0,
                alignment_loss=alignment_loss,
            )
        logits_for_dice = outputs["logits"] if isinstance(outputs, dict) else outputs
        if getattr(args, 'log_loss', False):
            for k, v in loss_dict.items():
                loss_accum[k] = loss_accum.get(k, 0.0) + v


        # ---- Centroid distance loss (all model types) ----
        if getattr(args, 'centroid', False):
            centroid_val = centroid_distance_loss(logits_for_dice, labels)
            total_loss_value = total_loss_value + args.centroid_weight * centroid_val
            if getattr(args, 'log_loss', False):
                loss_accum['loss_centroid'] = loss_accum.get('loss_centroid', 0.0) + centroid_val.item()
        
        # Backward pass
        optimizer.zero_grad()
        total_loss_value.backward()
        optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            dice = dice_coefficient(logits_for_dice, labels)
            if getattr(args, 'cldice', False):
                cldice_metric = cldice_coefficient(logits_for_dice, labels)
        
        total_loss += total_loss_value.item()
        total_dice += dice.item()
        if getattr(args, 'cldice', False):
            total_cldice += cldice_metric.item()
        num_batches += 1
        
        # Update progress bar
        postfix = {
            'loss': f'{total_loss_value.item():.4f}',
            'dice': f'{dice.item():.4f}'
        }
        if getattr(args, 'cldice', False):
            postfix['cldice'] = f'{cldice_metric.item():.4f}'
        pbar.set_postfix(postfix)
    
    avg_loss = total_loss / num_batches
    avg_dice = total_dice / num_batches
    avg_cldice = total_cldice / num_batches if getattr(args, 'cldice', False) else 0.0
    
    if getattr(args, 'cldice', False):
        log_msg = f"Train - Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}, clDice: {avg_cldice:.4f}\n"
    else:
        log_msg = f"Train - Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}\n"
    logger.write(log_msg)
    
    # --log_loss: Print epoch average for each loss component
    if getattr(args, 'log_loss', False) and loss_accum:
        parts = []
        for k in sorted(loss_accum.keys()):
            avg_v = loss_accum[k] / num_batches
            parts.append(f"{k}: {avg_v:.4f}")
        detail_msg = "  Loss detail - " + ", ".join(parts) + "\n"
        logger.write(detail_msg)
    
    # Return loss_accum averages as dict for JSONL logging
    loss_detail = {}
    if loss_accum:
        loss_detail = {k: v / num_batches for k, v in loss_accum.items()}
    
    return avg_loss, avg_dice, avg_cldice, loss_detail


def validate_epoch(model, val_loader, criterion, device, epoch, args, logger):
    """Validate one epoch. Metrics (Dice, clDice, HD95, ASSD, CD) are per-case mean (paper standard)."""
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    # Per-case lists for paper-standard metrics (mean over all validation samples)
    all_dice = []
    all_cldice = []
    all_hd95 = []
    all_assd = []
    all_centroid_dist = []

    has_dino = hasattr(model, 'dinov3_model') and model.dinov3_model is not None
    compute_cldice = getattr(args, 'cldice', False)
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            dino_slices = batch.get('dino_slices')
            if dino_slices is not None:
                dino_slices = dino_slices.to(device)
            
            mol_label = batch['mol'].to(device)
            if has_dino:
                outputs = model(images, mol_label=mol_label, dino_slices=dino_slices)
            else:
                outputs = model(images, mol_label=mol_label)

            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            loss = criterion(logits, labels)

            # Ensure logits match label spatial size (for per-sample metrics)
            if logits.shape[-2:] != labels.shape[-2:]:
                logits = F.interpolate(logits, size=labels.shape[-2:], mode='bilinear', align_corners=False)

            total_loss += loss.item()
            num_batches += 1

            # Per-case metrics (one value per sample, then mean over all samples)
            B = logits.shape[0]
            pred_np = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()
            label_np = labels.cpu().numpy()
            for b in range(B):
                all_dice.append(dice_coefficient(logits[b:b+1], labels[b:b+1]).item())
                if compute_cldice:
                    all_cldice.append(cldice_coefficient(logits[b:b+1], labels[b:b+1]).item())
                p_b = pred_np[b, 0]
                g_b = label_np[b, 0]
                all_hd95.append(compute_hd95(p_b, g_b))
                all_assd.append(compute_assd(p_b, g_b))
                all_centroid_dist.append(compute_centroid_distance(p_b, g_b))
    
    avg_loss = total_loss / max(num_batches, 1)
    n = len(all_dice)
    avg_dice = float(np.mean(all_dice)) if n else 0.0
    avg_cldice = float(np.mean(all_cldice)) if (compute_cldice and all_cldice) else 0.0
    avg_hd95 = float(np.nanmean(all_hd95)) if all_hd95 else 0.0
    avg_assd = float(np.nanmean(all_assd)) if all_assd else 0.0
    avg_centroid_dist = float(np.nanmean(all_centroid_dist)) if all_centroid_dist else 0.0
    
    if compute_cldice:
        log_msg = f"Val   - Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f} (per-case), clDice: {avg_cldice:.4f}, HD95: {avg_hd95:.2f}, ASSD: {avg_assd:.2f}, CD: {avg_centroid_dist:.2f}\n"
    else:
        log_msg = f"Val   - Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f} (per-case), HD95: {avg_hd95:.2f}, ASSD: {avg_assd:.2f}, CD: {avg_centroid_dist:.2f}\n"
    logger.write(log_msg)
    
    val_extra = {
        'hd95': avg_hd95,
        'assd': avg_assd,
        'centroid_dist': avg_centroid_dist,
    }
    
    return avg_loss, avg_dice, avg_cldice, val_extra


def save_checkpoint(model, optimizer, epoch, metrics, save_path, args,
                    scheduler=None, best_dice=0.0, best_epoch=0,
                    best_cldice=0.0, best_cldice_epoch=0):
    """Save model checkpoint (includes scheduler state for resume support)"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'args': vars(args),
        'best_dice': best_dice,
        'best_epoch': best_epoch,
        'best_cldice': best_cldice,
        'best_cldice_epoch': best_cldice_epoch,
    }
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, save_path)


def save_qualitative_results(
    model, val_loader, device, save_dir, epoch, args,
    max_samples=100, dice_threshold=0.7,
):
    """
    Saves validation results as 3-panel (Original / Prediction / GT) overlay images.
    
    - GT overlay: semi-transparent yellow
    - Prediction overlay: yellow (overlap with GT) + pink (False Positive)
    - If Dice < dice_threshold, appends '_low' to filename
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    has_dino = hasattr(model, 'dinov3_model') and model.dinov3_model is not None
    
    saved_count = 0
    
    with torch.no_grad():
        for batch in val_loader:
            if saved_count >= max_samples:
                break
            
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            patient_ids = batch['patient_id']
            slice_indices = batch['slice_idx']
            dino_slices = batch.get('dino_slices')
            if dino_slices is not None:
                dino_slices = dino_slices.to(device)
            
            mol_label = batch['mol'].to(device)
            if has_dino:
                outputs = model(images, mol_label=mol_label, dino_slices=dino_slices)
            else:
                outputs = model(images, mol_label=mol_label)
            
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            
            preds = (torch.sigmoid(logits) > 0.5).float()
            
            B = images.shape[0]
            for i in range(B):
                if saved_count >= max_samples:
                    break
                
                # Per-sample Dice
                pred_i = preds[i, 0].cpu().numpy()  # [H, W]
                gt_i = labels[i, 0].cpu().numpy()    # [H, W]
                
                inter = (pred_i * gt_i).sum()
                union = pred_i.sum() + gt_i.sum()
                dice_val = (2.0 * inter / (union + 1e-6)) if union > 0 else 0.0
                
                # Original image (first channel)
                img_i = images[i, 0].cpu().numpy()  # [H, W]
                # Normalize to [0, 1] for display
                img_min, img_max = img_i.min(), img_i.max()
                if img_max - img_min > 1e-8:
                    img_display = (img_i - img_min) / (img_max - img_min)
                else:
                    img_display = np.zeros_like(img_i)
                
                H, W = img_display.shape
                
                # ---- Create 3-panel figure ----
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                
                # Panel 1: Original image
                axes[0].imshow(img_display, cmap='gray', vmin=0, vmax=1)
                axes[0].set_title('Original', fontsize=14)
                axes[0].axis('off')
                
                # Panel 2: Prediction overlay on original
                axes[1].imshow(img_display, cmap='gray', vmin=0, vmax=1)
                # True positive (pred ∩ GT): yellow
                tp_mask = (pred_i > 0.5) & (gt_i > 0.5)
                # False positive (pred only): pink/light-red
                fp_mask = (pred_i > 0.5) & (gt_i < 0.5)
                # False negative (GT only, missed): bright orange
                fn_mask = (pred_i < 0.5) & (gt_i > 0.5)
                
                # Yellow overlay for TP
                if tp_mask.any():
                    overlay_tp = np.zeros((H, W, 4), dtype=np.float32)
                    overlay_tp[:, :, 0] = 1.0   # R
                    overlay_tp[:, :, 1] = 1.0   # G
                    overlay_tp[:, :, 2] = 0.0   # B
                    overlay_tp[:, :, 3] = tp_mask.astype(np.float32) * 0.5
                    axes[1].imshow(overlay_tp)
                
                # Pink overlay for FP
                if fp_mask.any():
                    overlay_fp = np.zeros((H, W, 4), dtype=np.float32)
                    overlay_fp[:, :, 0] = 1.0     # R
                    overlay_fp[:, :, 1] = 0.4     # G (pink tone)
                    overlay_fp[:, :, 2] = 0.6     # B
                    overlay_fp[:, :, 3] = fp_mask.astype(np.float32) * 0.6
                    axes[1].imshow(overlay_fp)
                
                # Bright orange overlay for FN (missed by prediction)
                if fn_mask.any():
                    overlay_fn = np.zeros((H, W, 4), dtype=np.float32)
                    overlay_fn[:, :, 0] = 1.0     # R
                    overlay_fn[:, :, 1] = 0.65    # G (orange tone)
                    overlay_fn[:, :, 2] = 0.0     # B
                    overlay_fn[:, :, 3] = fn_mask.astype(np.float32) * 0.55
                    axes[1].imshow(overlay_fn)
                
                axes[1].set_title(f'Prediction (Dice={dice_val:.3f})', fontsize=14)
                axes[1].axis('off')
                
                # Panel 3: GT overlay on original (yellow)
                axes[2].imshow(img_display, cmap='gray', vmin=0, vmax=1)
                if gt_i.any():
                    overlay_gt = np.zeros((H, W, 4), dtype=np.float32)
                    overlay_gt[:, :, 0] = 1.0   # R
                    overlay_gt[:, :, 1] = 1.0   # G
                    overlay_gt[:, :, 2] = 0.0   # B
                    overlay_gt[:, :, 3] = (gt_i > 0.5).astype(np.float32) * 0.5
                    axes[2].imshow(overlay_gt)
                
                axes[2].set_title('Ground Truth', fontsize=14)
                axes[2].axis('off')
                
                plt.tight_layout()
                
                # Filename
                pid = patient_ids[i]
                sidx = slice_indices[i]
                suffix = '_low' if dice_val < dice_threshold else ''
                fname = f'{pid}_slice{sidx:03d}_dice{dice_val:.3f}{suffix}.png'
                
                plt.savefig(save_dir / fname, dpi=120, bbox_inches='tight')
                plt.close(fig)
                
                saved_count += 1
    
    print(f"  Qualitative results saved: {saved_count} images -> {save_dir}")


def save_attention_maps(
    model, val_loader, device, save_dir, epoch, args,
    max_samples=100,
    train_loader=None,
):
    """
    Saves attention maps from AttentionPrototypeGenerator as jet colormap heatmaps.

    Save paths:
      {save_dir}/epoch{NN}_dice{X.XXXX}/val/{patient_id}_slice{NNN}.png
      {save_dir}/epoch{NN}_dice{X.XXXX}/train/{patient_id}_slice{NNN}.png  (if train_loader given)

    Format: [Predicted | GT Target] side-by-side (if GT is present)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image

    model.eval()
    save_dir = Path(save_dir)

    has_dino = hasattr(model, 'dinov3_model') and model.dinov3_model is not None

    def _collect_attn(loader, n):
        """Collect attention map data from a DataLoader."""
        data = []
        with torch.no_grad():
            for batch in loader:
                if len(data) >= n:
                    break

                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                patient_ids = batch['patient_id']
                slice_indices = batch['slice_idx']
                dino_slices = batch.get('dino_slices')
                if dino_slices is not None:
                    dino_slices = dino_slices.to(device)

                if hasattr(model, 'current_gt_mask'):
                    model.current_gt_mask = labels

                mol_label = batch['mol'].to(device)
                if has_dino:
                    outputs = model(images, mol_label=mol_label, dino_slices=dino_slices)
                else:
                    outputs = model(images, mol_label=mol_label)

                if isinstance(outputs, dict):
                    logits = outputs['logits']
                else:
                    logits = outputs

                if logits.shape[-2:] != labels.shape[-2:]:
                    logits = F.interpolate(logits, size=labels.shape[-2:], mode='bilinear', align_corners=False)

                # Prediction
                # Predicted cosine similarity map
                pred_sim = getattr(model, 'current_dino_sim_map', None)
                if pred_sim is None:
                    return data

                gt_sim = None
                if model.adapter is not None and hasattr(model.adapter, 'current_sim_map_gt'):
                    gt_sim = model.adapter.current_sim_map_gt

                H_img, W_img = images.shape[-2:]
                
                # Upsample 32x32 -> 512x512
                pred_sim_up = F.interpolate(pred_sim, size=(H_img, W_img), mode='bilinear', align_corners=False)
                gt_sim_up = None
                if gt_sim is not None:
                    gt_sim_up = F.interpolate(gt_sim, size=(H_img, W_img), mode='bilinear', align_corners=False)

                B = images.shape[0]
                for i in range(B):
                    if len(data) >= n:
                        break
                    
                    item = {
                        'attn': pred_sim_up[i, 0].cpu().numpy(),  # Save array for visualization
                        'img': images[i, 0].cpu().numpy(),
                        'pid': patient_ids[i],
                        'sidx': slice_indices[i],
                        'dice': 0.0, # (placeholder)
                    }
                    if gt_sim_up is not None:
                        item['attn_gt'] = gt_sim_up[i, 0].cpu().numpy()
                    data.append(item)
        return data

    def _to_jet(arr):
        """Normalize [H,W] array to [0,1] and apply jet colormap → uint8."""
        a_min, a_max = arr.min(), arr.max()
        if a_max - a_min > 1e-10:
            norm = (arr - a_min) / (a_max - a_min)
        else:
            norm = np.zeros_like(arr)
        colored = plt.cm.jet(norm)[:, :, :3]
        return (colored * 255).astype(np.uint8)

    def _save_split(data, out_dir, split_name):
        """Save collected attention data to disk."""
        if not data:
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        for item in data:
            pred_jet = _to_jet(item['attn'])
            pid = item['pid']
            sidx = item['sidx']
            fname = f'{pid}_slice{sidx:03d}.png'
            if 'attn_gt' in item:
                gt_jet = _to_jet(item['attn_gt'])
                composite = np.concatenate([pred_jet, gt_jet], axis=1)
                Image.fromarray(composite).save(out_dir / fname)
            else:
                Image.fromarray(pred_jet).save(out_dir / fname)
        has_gt = any('attn_gt' in d for d in data)
        fmt = "(Pred | GT)" if has_gt else "(Pred only)"
        print(f"    {split_name}: {len(data)} images {fmt} -> {out_dir}")

    # Collect from val
    val_data = _collect_attn(val_loader, max_samples)

    # Collect from train (if provided)
    train_data = []
    if train_loader is not None:
        train_data = _collect_attn(train_loader, max_samples)

    if not val_data and not train_data:
        print("  ⚠ No attention data collected.")
        return

    # Use val dice for directory name
    all_dice = [d['dice'] for d in val_data] if val_data else [d['dice'] for d in train_data]
    mean_dice = np.mean(all_dice)
    epoch_dir = save_dir / f'epoch{epoch:02d}_dice{mean_dice:.4f}'

    _save_split(val_data, epoch_dir / 'val', 'val')
    _save_split(train_data, epoch_dir / 'train', 'train')
    print(f"  Attention maps saved -> {epoch_dir}")

def main(args):
    # Set random seed
    seed_everything(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create experiment directory
    tz = pytz.timezone('Asia/Seoul')
    timestamp = datetime.now(tz).strftime('%Y_%m_%d_%H_%M_%S')
    model_name = getattr(args, 'model', 'unet')
    
    if args.exp_name:
        exp_name = args.exp_name
    else:
        # Reflect in name only if alpha/beta are explicitly passed in CLI
        cli_args = sys.argv[1:]
        has_alpha = '--adapter_loss_weight' in cli_args
        has_beta = '--point_loss_weight' in cli_args
        
        suffix_parts = []
        if has_alpha:
            alpha = getattr(args, 'adapter_loss_weight', 1.0)
            suffix_parts.append(f"alpha{alpha}")
        if has_beta:
            beta = getattr(args, 'point_loss_weight', 1.0)
            suffix_parts.append(f"beta{beta}")
        
        if suffix_parts:
            # If either alpha or beta is specified, include warm_up
            warmup = getattr(args, 'warm_up', 0)
            suffix = "_" + "_".join(suffix_parts) + f"_warmup{warmup}"
            exp_name = f"{model_name}_fold{args.fold}{suffix}"
        else:
            # If both are default, use only modelname_fold
            exp_name = f"{model_name}_fold{args.fold}"
    # Add '2.5d_' prefix to folder name if using 2.5D
    if args.use_2_5d:
        exp_name = f"2.5d_{exp_name}"
    exp_dir = Path(args.results_dir) / model_name / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create checkpoint directories
    ckpt_dir = exp_dir / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)
    
    if args.save_epochly:
        epoch_ckpt_dir = exp_dir / 'checkpoints_epoch'
        epoch_ckpt_dir.mkdir(exist_ok=True)
    
    # Setup logger
    log_file = exp_dir / f'training_log_{timestamp}.txt'
    logger = Logger(log_file)
    sys.stdout = logger
    
    print("=" * 80)
    print(f"Experiment: {exp_name}")
    print(f"Results directory: {exp_dir}")
    print("=" * 80)
    print(f"Arguments:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("=" * 80)
    
    # Initialize wandb
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=exp_name,
            config=vars(args)
        )
    
    # Load datasets
    print("\nLoading datasets...")
    target_size = (args.image_size, args.image_size) if args.image_size else None
    
    # Build train augmentation
    train_transforms = []
    if args.elastic:
        train_transforms.append(
            ElasticTransform(alpha=args.elastic_alpha, sigma=args.elastic_sigma, p=args.elastic_p)
        )
        print(f"  Augmentation: ElasticTransform(alpha={args.elastic_alpha}, sigma={args.elastic_sigma}, p={args.elastic_p})")
    if args.rotation:
        train_transforms.append(
            RandomRotation(degree_range=(args.rotation_min, args.rotation_max), p=args.rotation_p)
        )
        print(f"  Augmentation: RandomRotation(range=[{args.rotation_min}, {args.rotation_max}], p={args.rotation_p})")
    if args.zoom:
        train_transforms.append(
            RandomZoom(scale_range=(args.zoom_min, args.zoom_max), p=args.zoom_p)
        )
        print(f"  Augmentation: RandomZoom(range=[{args.zoom_min}, {args.zoom_max}], p={args.zoom_p})")
    train_transform = ComposeTransforms(train_transforms) if train_transforms else None
    
    train_dataset = TMJ_dataset_2D(
        metadata_file=args.metadata_file,
        data_dir=args.data_dir,
        label_dir=args.label_dir,
        split='train',
        fold=args.fold,
        val_ratio=args.val_ratio,
        fold_seed=args.seed,
        normalize=args.normalize,
        target_size=target_size,
        splits_json_path=args.splits_json if args.splits_json else None,
        use_2_5d=args.use_2_5d,
        transform=train_transform,
    )
    
    val_dataset = TMJ_dataset_2D(
        metadata_file=args.metadata_file,
        data_dir=args.data_dir,
        label_dir=args.label_dir,
        split='val',
        fold=args.fold,
        val_ratio=args.val_ratio,
        fold_seed=args.seed,
        normalize=args.normalize,
        target_size=target_size,
        splits_json_path=args.splits_json if args.splits_json else None,
        use_2_5d=args.use_2_5d,
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn_2d,
        worker_init_fn=seed_worker
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn_2d
    )
    
    # Build model
    print("\nBuilding model...")
    print(f"Model type: {args.model}")
    
    # Load MedDINOv3 (required for learnable-query model)
    dinov3_model = None
    print("Loading MedDINOv3...")
    try:
        meddinov3_path = Path("/path/to/MedDINOv3")
        if str(meddinov3_path) not in sys.path:
            sys.path.insert(0, str(meddinov3_path))
        from dinov3.models.vision_transformer import vit_base

        dinov3_model = vit_base(
            patch_size=16,
            img_size=518,
            init_values=1.0,
            block_chunks=0,
            num_register_tokens=0,
            interpolate_antialias=False,
            interpolate_offset=0.1
        )

        dinov3_checkpoint = "/path/to/MedDINOv3/checkpoint/model.pth"
        if os.path.exists(dinov3_checkpoint):
            checkpoint = torch.load(dinov3_checkpoint, map_location='cpu')
            state_dict = checkpoint.get('teacher', checkpoint.get('model', checkpoint))
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k
                if new_key.startswith('module.'):
                    new_key = new_key[7:]
                if new_key.startswith('backbone.'):
                    new_key = new_key[9:]
                new_state_dict[new_key] = v
            load_result = dinov3_model.load_state_dict(new_state_dict, strict=False)
            n_matched = len(set(dinov3_model.state_dict().keys()) & set(new_state_dict.keys()))
            print(f"✓ MedDINOv3 loaded from {dinov3_checkpoint} (matched {n_matched}/{len(dinov3_model.state_dict())} keys)")
            if load_result.missing_keys:
                print(f"  Missing keys: {len(load_result.missing_keys)}")

        dinov3_model.eval()
        for p in dinov3_model.parameters():
            p.requires_grad = False
        dinov3_model = dinov3_model.to(device)
        print("✓ MedDINOv3 frozen and ready")
    except Exception as e:
        print(f"⚠ Failed to load MedDINOv3: {e}")
        print("  Training without MedDINOv3")
        dinov3_model = None
    
    # Create model
    enable_point_rend = args.warm_up > 0  # PointRend is activated when warm_up > 0
    
    model = get_model(
        args.model,
        n_channels=args.n_channels,
        n_classes=args.n_classes,
        bilinear=args.bilinear,
        dinov3_model=dinov3_model,
        dinov3_feature_dim=768,
        dinov3_input_size=518,
        enable_point_rend=enable_point_rend,
        main_slice_weight=getattr(args, 'main_slice_weight', 1.0),
        learnable_query_ckpt=getattr(args, 'learnable_query_ckpt', ''),
    )
    model = model.to(device)
    
    # Load pretrained weights (only one checkpoint option can be used)
    if args.resume and (args.unet_checkpoint or args.checkpoint_unet_prototype):
        raise ValueError("--resume cannot be used together with --unet_checkpoint or --checkpoint_unet_prototype.")
    if args.unet_checkpoint and args.checkpoint_unet_prototype:
        raise ValueError("Cannot use both --unet_checkpoint and --checkpoint_unet_prototype. Choose one.")
    
    # Load pretrained UNet weights if provided (PointRend modules remain randomly initialized)
    if args.unet_checkpoint:
        print(f"\nLoading pretrained UNet weights from: {args.unet_checkpoint}")
        try:
            checkpoint = torch.load(args.unet_checkpoint, map_location=device)
            if 'model_state_dict' in checkpoint:
                pretrained_dict = checkpoint['model_state_dict']
            else:
                pretrained_dict = checkpoint
            
            # Filter: Load UNet backbone only (exclude PointRend, DINO)
            model_dict = model.state_dict()
            filtered_dict = {
                k: v for k, v in pretrained_dict.items()
                if k in model_dict 
                and not k.startswith('coarse_head') 
                and not k.startswith('point_head')
                and not k.startswith('adapter')
                and not k.startswith('dino_proj')
            }
            
            # Print load info
            loaded_keys = set(filtered_dict.keys())
            all_keys = set(model_dict.keys())
            not_loaded = all_keys - loaded_keys
            
            model_dict.update(filtered_dict)
            model.load_state_dict(model_dict)
            
            print(f"✓ Loaded {len(filtered_dict)} layers from checkpoint (UNet backbone only)")
            print(f"✓ Randomly initialized: PointRend (coarse_head, point_head), DINO (adapter, dino_proj)")
        except Exception as e:
            print(f"⚠ Failed to load checkpoint: {e}")
            print("  Proceeding with random initialization")
    
    # Load pretrained UNet+DINO+Adapter weights (PointRend modules remain randomly initialized)
    if args.checkpoint_unet_prototype:
        print(f"\nLoading pretrained UNet+DINO+Adapter weights from: {args.checkpoint_unet_prototype}")
        try:
            checkpoint = torch.load(args.checkpoint_unet_prototype, map_location=device)
            if 'model_state_dict' in checkpoint:
                pretrained_dict = checkpoint['model_state_dict']
            else:
                pretrained_dict = checkpoint
            
            # Filter: Load UNet backbone + DINO adapter (exclude PointRend, shape mismatch)
            model_dict = model.state_dict()
            filtered_dict = {
                k: v for k, v in pretrained_dict.items()
                if k in model_dict 
                and not k.startswith('coarse_head') 
                and not k.startswith('point_head')
                and v.shape == model_dict[k].shape
            }
            
            # Print load info
            loaded_keys = set(filtered_dict.keys())
            all_keys = set(model_dict.keys())
            not_loaded = all_keys - loaded_keys
            
            model_dict.update(filtered_dict)
            model.load_state_dict(model_dict)
            
            print(f"✓ Loaded {len(filtered_dict)} layers from checkpoint")
            print(f"  - Encoder, Decoder (up1~up4), DINO (adapter, dino_proj): from checkpoint")
            print(f"  - PointRend (coarse_head, point_head): randomly initialized")
            pointrend_keys = [k for k in not_loaded if 'coarse_head' in k or 'point_head' in k]
            if pointrend_keys:
                print(f"✓ Randomly initialized layers: {pointrend_keys[:5]}{'...' if len(pointrend_keys) > 5 else ''}")
        except Exception as e:
            print(f"⚠ Failed to load checkpoint: {e}")
            print("  Proceeding with random initialization")
    
    # Freeze UNet backbone if requested
    if args.freeze_unet:
        print(f"\n⚠ Freezing UNet backbone (encoder + decoder up1~up2)")
        freeze_count = 0
        for name, param in model.named_parameters():
            # Freeze: encoder (inc, down1~4) + decoder (up1, up2, up3, up4, outc)
            # Not freeze: coarse_head, point_head, adapter, dino_proj
            if not any(x in name for x in ['coarse_head', 'point_head', 'adapter', 'dino_proj']):
                param.requires_grad = False
                freeze_count += 1
        print(f"✓ Frozen {freeze_count} parameter groups (UNet backbone)")
        print(f"✓ Training only: PointRend modules (coarse_head, point_head) + adapter/dino_proj (if exists)")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    if enable_point_rend:
        print(f"PointRend enabled: warm_up={args.warm_up}, point_loss_weight={args.point_loss_weight}")
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs, 
        eta_min=args.min_lr
    )
    
    # Setup loss function
    if args.loss_type == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss_type == 'dice':
        criterion = dice_loss
    elif args.loss_type == 'combined':
        criterion = lambda pred, target: combined_loss(pred, target, args.bce_weight, args.dice_weight)
    else:
        raise ValueError(f"Unknown loss type: {args.loss_type}")
    
    # ---- Resume from checkpoint ----
    start_epoch = 1
    best_dice = 0.0
    best_epoch = 0
    best_cldice = 0.0
    best_cldice_epoch = 0
    
    if args.resume:
        print(f"\n{'=' * 80}")
        print(f"Resuming from checkpoint: {args.resume}")
        assert os.path.isfile(args.resume), f"Checkpoint not found: {args.resume}"
        ckpt = torch.load(args.resume, map_location=device)
        
        # Model weights (full)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"  ✓ Model weights restored")
        
        # Optimizer state
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        print(f"  ✓ Optimizer state restored")
        
        # Scheduler state
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            print(f"  ✓ Scheduler state restored from checkpoint")
        else:
            # Manually step scheduler since legacy checkpoints lack state_dict
            resumed_epoch = ckpt['epoch']
            for _ in range(resumed_epoch):
                scheduler.step()
            print(f"  ⚠ Scheduler state not in checkpoint — manually stepped {resumed_epoch} times")
        
        # Epoch
        start_epoch = ckpt['epoch'] + 1
        print(f"  ✓ Resuming from epoch {start_epoch}")
        
        # Best metrics
        best_dice = ckpt.get('best_dice', ckpt.get('metrics', {}).get('val_dice', 0.0))
        best_epoch = ckpt.get('best_epoch', ckpt.get('epoch', 0))
        best_cldice = ckpt.get('best_cldice', 0.0)
        best_cldice_epoch = ckpt.get('best_cldice_epoch', 0)
        print(f"  ✓ Best dice so far: {best_dice:.4f} (epoch {best_epoch})")
        if best_cldice > 0:
            print(f"  ✓ Best clDice so far: {best_cldice:.4f} (epoch {best_cldice_epoch})")
        print(f"{'=' * 80}")
    
    # Training loop
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 80)
        if args.warm_up > 0 and epoch == args.warm_up + 1:
            print(">>> PointRend enabled from this epoch (warmup finished).")
        
        # Train
        train_loss, train_dice, train_cldice, train_loss_detail = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args, logger
        )
        
        # Validate
        val_loss, val_dice, val_cldice, val_extra = validate_epoch(
            model, val_loader, criterion, device, epoch, args, logger
        )
        val_hd95 = val_extra['hd95']
        val_assd = val_extra['assd']
        val_cd = val_extra['centroid_dist']
        
        # Learning rate step
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        # Log to wandb
        if args.wandb:
            wandb_dict = {
                'epoch': epoch,
                'train/loss': train_loss,
                'train/dice': train_dice,
                'val/loss': val_loss,
                'val/dice': val_dice,
                'val/hd95': val_hd95,
                'val/assd': val_assd,
                'val/centroid_dist': val_cd,
                'lr': current_lr
            }
            if getattr(args, 'cldice', False):
                wandb_dict['train/cldice'] = train_cldice
                wandb_dict['val/cldice'] = val_cldice
            wandb.log(wandb_dict)
        
        # Save best checkpoint
        if val_dice > best_dice:
            best_dice = val_dice
            best_epoch = epoch
            
            best_path = ckpt_dir / f'best_epoch{epoch:03d}_{val_dice:.4f}.pth'
            save_checkpoint(
                model, optimizer, epoch,
                {'val_loss': val_loss, 'val_dice': val_dice},
                best_path, args,
                scheduler=scheduler, best_dice=best_dice, best_epoch=best_epoch,
                best_cldice=best_cldice, best_cldice_epoch=best_cldice_epoch,
            )
            print(f"✓ Saved best checkpoint: dice={val_dice:.4f}")
        
        # Save best clDice checkpoint (saved separately)
        if getattr(args, 'cldice', False) and val_cldice > best_cldice:
            best_cldice = val_cldice
            best_cldice_epoch = epoch
            
            best_cldice_path = ckpt_dir / f'best_cldice_epoch{epoch:03d}_{val_cldice:.4f}.pth'
            save_checkpoint(
                model, optimizer, epoch,
                {'val_loss': val_loss, 'val_dice': val_dice, 'val_cldice': val_cldice},
                best_cldice_path, args,
                scheduler=scheduler, best_dice=best_dice, best_epoch=best_epoch,
                best_cldice=best_cldice, best_cldice_epoch=best_cldice_epoch,
            )
            print(f"✓ Saved best clDice checkpoint: cldice={val_cldice:.4f}")
        
        # JSONL logging — always enabled (after best_dice update)
        jsonl_path = exp_dir / 'train_log.jsonl'
        log_entry = {
            'epoch': epoch,
            'train_loss': round(train_loss, 6),
            'train_dice': round(train_dice, 6),
            'val_loss': round(val_loss, 6),
            'val_dice': round(val_dice, 6),
            'lr': round(current_lr, 8),
            'best_dice': round(best_dice, 6),
            'best_epoch': best_epoch,
        }
        # Add per-component loss detail
        if train_loss_detail:
            for k, v in train_loss_detail.items():
                log_entry[f'train_{k}'] = round(v, 6)
        # Add clDice metrics
        if getattr(args, 'cldice', False):
            log_entry['train_cldice'] = round(train_cldice, 6)
            log_entry['val_cldice'] = round(val_cldice, 6)
            log_entry['best_cldice'] = round(best_cldice, 6)
            log_entry['best_cldice_epoch'] = best_cldice_epoch
        # Add surface distance metrics
        log_entry['val_hd95'] = round(val_hd95, 4)
        log_entry['val_assd'] = round(val_assd, 4)
        log_entry['val_centroid_dist'] = round(val_cd, 4)
        with open(jsonl_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # Save latest checkpoint
        latest_path = ckpt_dir / 'latest.pth'
        save_checkpoint(
            model, optimizer, epoch,
            {'val_loss': val_loss, 'val_dice': val_dice},
            latest_path, args,
            scheduler=scheduler, best_dice=best_dice, best_epoch=best_epoch,
            best_cldice=best_cldice, best_cldice_epoch=best_cldice_epoch,
        )
        
        # Save epoch checkpoint
        if args.save_epochly:
            epoch_path = epoch_ckpt_dir / f'epoch_{epoch:03d}_{val_dice:.4f}.pth'
            save_checkpoint(
                model, optimizer, epoch,
                {'val_loss': val_loss, 'val_dice': val_dice},
                epoch_path, args,
                scheduler=scheduler, best_dice=best_dice, best_epoch=best_epoch,
                best_cldice=best_cldice, best_cldice_epoch=best_cldice_epoch,
            )
        
        print(f"Current LR: {current_lr:.6f}")
        print(f"Best dice so far: {best_dice:.4f} (epoch {best_epoch})")
        if getattr(args, 'cldice', False):
            print(f"Best clDice so far: {best_cldice:.4f} (epoch {best_cldice_epoch})")
        
        # Save qualitative results at specified interval
        if (
            getattr(args, 'save_qualitative_epoch', 0) > 0
            and epoch % args.save_qualitative_epoch == 0
        ):
            qual_dir = exp_dir / 'qualitative' / f'epoch_{epoch:03d}'
            print(f"  Saving qualitative results for epoch {epoch}...")
            save_qualitative_results(
                model, val_loader, device, qual_dir, epoch, args,
                max_samples=100, dice_threshold=0.7,
            )
        
        # Save attention maps at specified interval
        if (
            getattr(args, 'save_attn_epoch', 0) > 0
            and epoch % args.save_attn_epoch == 0
            and hasattr(model, 'current_attn_weights')
        ):
            attn_save_dir = exp_dir / 'attn_map'
            print(f"  Saving attention maps for epoch {epoch}...")
            save_attention_maps(
                model, val_loader, device, attn_save_dir, epoch, args,
                max_samples=getattr(args, 'save_attn_max', 100),
                train_loader=train_loader,
            )
        
        print("=" * 80)
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best validation dice: {best_dice:.4f} at epoch {best_epoch}")
    if getattr(args, 'cldice', False):
        print(f"Best validation clDice: {best_cldice:.4f} at epoch {best_cldice_epoch}")
    print(f"Results saved to: {exp_dir}")
    print("=" * 80)
    
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='UNet Training for TMJ 2D Segmentation')
    
    # Dataset arguments
    parser.add_argument('--metadata_file', type=str,
                        default='/path/to/data.csv',
                        help='Path to metadata CSV file')
    parser.add_argument('--data_dir', type=str,
                        default='/path/to/data_converted/nifti',
                        help='Path to converted per-slice .npy directory')
    parser.add_argument('--label_dir', type=str,
                        default='/path/to/data/labels',
                        help='Path to label directory')
    parser.add_argument('--fold', type=int, default=0,
                        help='Fold index (0-4). Used with --splits_json for train/val split.')
    parser.add_argument('--splits_json', type=str,
                        default='/path/to/splits_final.json',
                        help='Path to nnUNet splits_final.json. Train/val use this split. Set empty to use CSV fold instead.')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Validation ratio when using fold-based split')
    parser.add_argument('--normalize', type=str, default='minmax', choices=['minmax', 'zscore', 'none'],
                        help='Normalization method')
    parser.add_argument('--image_size', type=int, default=512,
                        help='Resize image/label to (image_size, image_size). Set 0 to disable (may cause batch size mismatch).')
    parser.add_argument('--use_2_5d', action='store_true',
                        help='Use 2.5D input: [prev, curr, next] slices as 3 channels (instead of replicating single slice)')
    parser.add_argument('--elastic', action='store_true',
                        help='Apply elastic deformation augmentation during training (alpha=80, sigma=8, p=0.5)')
    parser.add_argument('--elastic_alpha', type=float, default=80.0,
                        help='Elastic deformation alpha (magnitude). Default: 80')
    parser.add_argument('--elastic_sigma', type=float, default=8.0,
                        help='Elastic deformation sigma (smoothness). Default: 8')
    parser.add_argument('--elastic_p', type=float, default=0.5,
                        help='Elastic deformation probability. Default: 0.5')
    parser.add_argument('--rotation', action='store_true',
                        help='Apply random rotation augmentation during training')
    parser.add_argument('--rotation_min', type=float, default=-15.0,
                        help='Rotation minimum angle in degrees. Default: -15')
    parser.add_argument('--rotation_max', type=float, default=15.0,
                        help='Rotation maximum angle in degrees. Default: 15')
    parser.add_argument('--rotation_p', type=float, default=0.5,
                        help='Rotation probability. Default: 0.5')
    parser.add_argument('--zoom', action='store_true',
                        help='Apply random zoom augmentation during training (center crop after scale)')
    parser.add_argument('--zoom_min', type=float, default=0.0,
                        help='Zoom minimum scale fraction (0.0 = 1.0x). Default: 0.0')
    parser.add_argument('--zoom_max', type=float, default=0.1,
                        help='Zoom maximum scale fraction (0.1 = 1.1x). Default: 0.1')
    parser.add_argument('--zoom_p', type=float, default=0.5,
                        help='Zoom probability. Default: 0.5')
    
    # Model arguments
    parser.add_argument(
        '--model',
        type=str,
        default='unet_all_mol_learnable_query',
        help='Model type (only unet_all_mol_learnable_query is supported)',
    )
    parser.add_argument(
        '--n_channels',
        type=int,
        default=3,
        help=(
            'Number of input channels '
            '(1 or 3 for unet/unet_pointrend, 3 for unet_all_protopype_pointrend)'
        ),
    )
    parser.add_argument('--n_classes', type=int, default=1,
                        help='Number of output classes')
    parser.add_argument('--bilinear', action='store_true',
                        help='Use bilinear upsampling instead of transposed convolution')
    parser.add_argument('--warm_up', type=int, default=0,
                        help='PointRend warm-up epochs (0=disabled, >0=enable after this epoch)')
    parser.add_argument('--adapter_loss_weight', type=float, default=1.0,
                        help='Weight for adapter alignment loss (alpha)')
    parser.add_argument('--point_loss_weight', type=float, default=1.0,
                        help='Weight for point refinement loss (beta)')
    parser.add_argument('--unet_checkpoint', type=str, default=None,
                        help='Path to pretrained UNet checkpoint (only backbone weights loaded, PointRend modules init from scratch)')
    parser.add_argument('--checkpoint_unet_prototype', type=str, default=None,
                        help='Path to pretrained UNet+DINO+Adapter checkpoint (model_unet_prototype.py). Loads encoder, decoder, adapter, dino_proj. PointRend modules init from scratch.')
    parser.add_argument('--freeze_unet', action='store_true',
                        help='Freeze UNet backbone (encoder + decoder up1~up2), only train PointRend modules')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from. '
                             'Restores model weights, optimizer state, scheduler state, epoch, and best metrics. '
                             'Overrides --unet_checkpoint / --checkpoint_unet_prototype if set.')
    parser.add_argument('--main_slice_weight', type=float, default=1.0,
                        help='Weight for the main (center) slice when averaging DINO features across adjacent slices. '
                             'Adjacent slices always get weight 1. Result = (prev + w*curr + next) / (w+2). '
                             'Default: 1.0 (equal averaging = simple mean).')
    
    parser.add_argument('--learnable_query_ckpt', type=str, default='',
                        help='Path to pretrained CrossAttentionQueryExtractor checkpoint')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--val_batch_size', type=int, default=32,
                        help='Validation batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate (default: 1e-3 for Adam)')
    parser.add_argument('--min_lr', type=float, default=1e-5,
                        help='Minimum learning rate for cosine scheduler (default: 1e-5, i.e., lr/100)')
    # parser.add_argument('--lr', type=float, default=1e-2,
    #                     help='Initial learning rate (default: 1e-3 for Adam)')
    # parser.add_argument('--min_lr', type=float, default=1e-4,
    #                     help='Minimum learning rate for cosine scheduler (default: 1e-5, i.e., lr/100)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--loss_type', type=str, default='combined',
                        choices=['bce', 'dice', 'combined'],
                        help='Loss function type')
    parser.add_argument('--bce_weight', type=float, default=0.5,
                        help='Weight for BCE loss in combined loss')
    parser.add_argument('--dice_weight', type=float, default=0.5,
                        help='Weight for Dice loss in combined loss')
    parser.add_argument('--coarse_weight', type=float, default=0.5,
                        help='Weight for coarse loss (default: 0.5)')
    
    parser.add_argument('--cldice', action='store_true',
                        help='Compute clDice (centerline Dice) metric using soft-skeletonization for evaluation')
    parser.add_argument('--centroid', action='store_true',
                        help='Add centroid distance loss — penalizes displacement between predicted and GT mask centroids')
    parser.add_argument('--centroid_weight', type=float, default=1.0,
                        help='Weight for centroid distance loss (default: 1.0)')
    parser.add_argument('--distill_weight', type=float, default=1.0,
                        help='Weight for spatial consistency loss L_distill (default: 1.0, only for unet_all_mol_coarseDINO)')
    
    # System arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Experiment arguments
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name (auto-generated if not provided)')
    parser.add_argument('--results_dir', type=str,
                        default='./results/UNet',
                        help='Results directory')
    parser.add_argument('--save_epochly', action='store_true',
                        help='Save checkpoint for each epoch')
    parser.add_argument('--log_loss', action='store_true', default=True,
                        help='Log per-component loss breakdown each epoch + JSONL. Enabled by default.')
    parser.add_argument('--save_qualitative_epoch', type=int, default=0,
                        help='Save 3-panel qualitative results (Original/Prediction/GT) every N epochs. '
                             '0=disabled, e.g. 10 saves at epoch 10, 20, 30, ...')
    parser.add_argument('--save_attn_epoch', type=int, default=0,
                        help='Save attention maps from AttentionPrototypeGenerator every N epochs. '
                             '0=disabled. Also saves at epoch 0. (unet_all_mol_2 only)')
    parser.add_argument('--save_attn_max', type=int, default=100,
                        help='Maximum number of attention map images to save per epoch (default: 100)')
    
    # Wandb arguments
    parser.add_argument('--wandb', action='store_true',
                        help='Use Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='TMJ_UNet_2D',
                        help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default='micv-uss24',
                        help='Wandb entity name')
    
    args = parser.parse_args()
    
    # Validate: use_2_5d requires n_channels=3
    if args.use_2_5d and args.n_channels != 3:
        raise ValueError(
            f"--use_2_5d requires --n_channels=3 (got {args.n_channels}). "
            "2.5D mode stacks [prev, curr, next] slices as 3 channels."
        )
    
    main(args)
