"""
UNet Validation/Inference Script for TMJ 2D Segmentation (npy slice data)

Uses dataset_2d_slice.py (pre-converted .npy slices) for input.
Supports all model types including MOL-conditional models.
Computes Dice, clDice, IoU, Precision, Recall metrics.

# Save predictions
python validate_slice.py \
    --fold 0 \
    --checkpoint results/UNet/.../checkpoints/best.pth \
    --save_predictions \
    --output_dir validation_fold0
"""

import os
import sys
import json
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
from PIL import Image
from scipy.ndimage import distance_transform_edt, binary_erosion

from dataset_2d_slice import TMJ_dataset_2D, collate_fn_2d
from models import get_model


# ============================================================================
# Metric Functions
# ============================================================================

def dice_coefficient(pred, target, smooth=1e-6):
    """
    Calculate Dice coefficient.
    
    Args:
        pred: Predicted logits [B, 1, H, W]
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


def iou_score(pred, target, smooth=1e-6):
    """Calculate IoU score."""
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou


def precision_score(pred, target, smooth=1e-6):
    """Calculate precision."""
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    true_positive = (pred * target).sum()
    predicted_positive = pred.sum()
    
    precision = (true_positive + smooth) / (predicted_positive + smooth)
    return precision


def recall_score(pred, target, smooth=1e-6):
    """Calculate recall."""
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    true_positive = (pred * target).sum()
    actual_positive = target.sum()
    
    recall = (true_positive + smooth) / (actual_positive + smooth)
    return recall


# ── clDice (Soft-Skeletonization) ──

def soft_erode(img):
    """Soft erosion using min pool."""
    if len(img.shape) == 4:
        p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
        return torch.min(p1, p2)
    else:
        p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
        p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
        p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
        return torch.min(torch.min(p1, p2), p3)


def soft_dilate(img):
    """Soft dilation using max pool."""
    if len(img.shape) == 4:
        return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))
    else:
        return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))


def soft_open(img):
    """Soft opening = erode then dilate."""
    return soft_dilate(soft_erode(img))


def soft_skeletonize(img, iters=10):
    """Iterative soft-skeletonization."""
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iters):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def cldice_coefficient(pred, target, iters=10, smooth=1e-6):
    """
    Calculate clDice (centerline Dice) metric.
    
    Args:
        pred: Predicted logits [B, 1, H, W]
        target: Ground truth mask [B, 1, H, W]
    
    Returns:
        clDice value (scalar)
    """
    pred_prob = torch.sigmoid(pred)
    pred_binary = (pred_prob > 0.5).float()
    
    skel_pred = soft_skeletonize(pred_binary, iters=iters)
    skel_target = soft_skeletonize(target, iters=iters)
    
    tprec = ((skel_pred * target).sum() + smooth) / (skel_pred.sum() + smooth)
    tsens = ((skel_target * pred_binary).sum() + smooth) / (skel_target.sum() + smooth)
    
    cldice = 2.0 * tprec * tsens / (tprec + tsens + smooth)
    return cldice


def compute_hd95(pred_binary_np, target_np):
    """
    95% Hausdorff Distance (HD95) for a single 2D binary mask pair.
    Returns np.nan if either mask is empty.
    """
    pred_bool = pred_binary_np.astype(bool)
    target_bool = target_np.astype(bool)
    
    if not pred_bool.any() or not target_bool.any():
        return np.nan
    
    dt_target = distance_transform_edt(~target_bool)
    dt_pred = distance_transform_edt(~pred_bool)
    
    pred_boundary = pred_bool ^ binary_erosion(pred_bool)
    target_boundary = target_bool ^ binary_erosion(target_bool)
    
    if not pred_boundary.any() or not target_boundary.any():
        return np.nan
    
    dist_pred_to_gt = dt_target[pred_boundary]
    dist_gt_to_pred = dt_pred[target_boundary]
    
    all_distances = np.concatenate([dist_pred_to_gt, dist_gt_to_pred])
    return np.percentile(all_distances, 95)


def compute_assd(pred_binary_np, target_np):
    """
    Average Symmetric Surface Distance (ASSD) for a single 2D binary mask pair.
    Returns np.nan if either mask is empty.
    """
    pred_bool = pred_binary_np.astype(bool)
    target_bool = target_np.astype(bool)
    
    if not pred_bool.any() or not target_bool.any():
        return np.nan
    
    dt_target = distance_transform_edt(~target_bool)
    dt_pred = distance_transform_edt(~pred_bool)
    
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
    Returns np.nan if either mask is empty.
    """
    pred_bool = pred_binary_np.astype(bool)
    target_bool = target_np.astype(bool)
    
    if not pred_bool.any() or not target_bool.any():
        return np.nan
    
    pred_coords = np.argwhere(pred_bool)
    gt_coords = np.argwhere(target_bool)
    
    pred_centroid = pred_coords.mean(axis=0)
    gt_centroid = gt_coords.mean(axis=0)
    
    return np.sqrt(((pred_centroid - gt_centroid) ** 2).sum())


# ============================================================================
# Validate Function
# ============================================================================

def validate(model, val_loader, device, args, save_predictions=False, save_masks=False, output_dir=None):
    """
    Validate model on validation set.
    
    Args:
        model: Model
        val_loader: Validation data loader
        device: Device to run validation on
        args: Command-line arguments (for model type info)
        save_predictions: Whether to save prediction masks
        output_dir: Directory to save predictions
    
    Returns:
    Returns:
        avg_metrics, patient_avg_metrics, case_results
    """
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    # Incremental accumulators for global metrics (memory-efficient)
    # Per-class TP/FP/FN for global Dice
    global_tp = np.zeros(2, dtype=np.float64)  # [bg, fg]
    global_fp = np.zeros(2, dtype=np.float64)
    global_fn = np.zeros(2, dtype=np.float64)
    # Per-class IoU accumulators
    global_inter = np.zeros(2, dtype=np.float64)
    global_union = np.zeros(2, dtype=np.float64)
    # Per-sample Dice list (lightweight: just scalars)
    all_dice_per_sample = []
    
    # Per-patient metrics
    patient_metrics = {}
    
    # Per-case results (for CSV)
    case_results = []
    
    # Queue for deferred prediction saving (separate tqdm)
    save_queue = []
    
    criterion = nn.BCEWithLogitsLoss()
    
    # MOL learnable-query model forward
    has_dino = hasattr(model, 'dinov3_model') and model.dinov3_model is not None
    compute_cldice = getattr(args, 'cldice', True)  # default enabled for validation
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validating")):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            patient_ids = batch['patient_id']
            slice_indices = batch['slice_idx']
            dino_slices = batch.get('dino_slices')
            if dino_slices is not None:
                dino_slices = dino_slices.to(device)
            
            mol_label = batch['mol'].to(device)
            if hasattr(model, 'current_gt_mask'):
                model.current_gt_mask = labels
            if has_dino:
                outputs = model(images, mol_label=mol_label, dino_slices=dino_slices)
            else:
                outputs = model(images, mol_label=mol_label)
            
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            
            # Ensure logits match label spatial size
            if logits.shape[-2:] != labels.shape[-2:]:
                logits = F.interpolate(logits, size=labels.shape[-2:], mode='bilinear', align_corners=False)
            
            # Calculate loss
            loss = criterion(logits, labels)
            total_loss += loss.item()
            num_batches += 1
            
            # Convert to binary predictions
            preds = torch.sigmoid(logits)
            preds_binary = (preds > 0.5).float()
            
            # Accumulate global metrics incrementally (no large array storage)
            pred_np = preds_binary.cpu().numpy()[:, 0]  # [B, H, W]
            label_np = labels.cpu().numpy()[:, 0]        # [B, H, W]
            for c in range(2):
                pc = (pred_np == c)
                lc = (label_np == c)
                global_tp[c] += (pc & lc).sum()
                global_fp[c] += (pc & ~lc).sum()
                global_fn[c] += (~pc & lc).sum()
                inter_c = (pc & lc).sum()
                union_c = pc.sum() + lc.sum() - inter_c
                global_inter[c] += inter_c
                global_union[c] += union_c
            # Per-sample Dice
            for b in range(pred_np.shape[0]):
                p = pred_np[b].flatten()
                l = label_np[b].flatten()
                inter_s = (p * l).sum()
                union_s = p.sum() + l.sum()
                d = 2 * inter_s / union_s if union_s > 0 else 0.0
                all_dice_per_sample.append(d)
            
            # Calculate per-sample metrics
            for i in range(len(patient_ids)):
                pid = patient_ids[i]
                
                if pid not in patient_metrics:
                    patient_metrics[pid] = {
                        'dice': [],
                        'iou': [],
                        'voe': [],
                        'precision': [],
                        'recall': [],
                        'cldice': [],
                        'hd95': [],
                        'assd': [],
                        'centroid_dist': [],
                    }
                
                single_output = logits[i:i+1]
                single_label = labels[i:i+1]
                
                patient_metrics[pid]['dice'].append(dice_coefficient(single_output, single_label).item())
                iou_val = iou_score(single_output, single_label).item()
                patient_metrics[pid]['iou'].append(iou_val)
                patient_metrics[pid]['voe'].append(1.0 - iou_val)
                patient_metrics[pid]['precision'].append(precision_score(single_output, single_label).item())
                patient_metrics[pid]['recall'].append(recall_score(single_output, single_label).item())
                if compute_cldice:
                    patient_metrics[pid]['cldice'].append(cldice_coefficient(single_output, single_label).item())
                
                # Surface distance metrics (on CPU numpy)
                pred_bin_np = (torch.sigmoid(single_output) > 0.5).float().cpu().numpy()[0, 0]
                label_np_single = single_label.cpu().numpy()[0, 0]
                patient_metrics[pid]['hd95'].append(compute_hd95(pred_bin_np, label_np_single))
                patient_metrics[pid]['assd'].append(compute_assd(pred_bin_np, label_np_single))
                patient_metrics[pid]['centroid_dist'].append(compute_centroid_distance(pred_bin_np, label_np_single))
                
                # Collect case details
                meta = batch['metadata'][i] if 'metadata' in batch else {}
                label_path = meta.get('label_path', 'unknown')
                filename = Path(label_path).name
                
                case_entry = {
                    'filename': filename,
                    'patient_id': pid,
                    'slice_idx': slice_indices[i].item() if isinstance(slice_indices[i], torch.Tensor) else slice_indices[i],
                    'dice': patient_metrics[pid]['dice'][-1],
                    'iou': patient_metrics[pid]['iou'][-1],
                    'voe': patient_metrics[pid]['voe'][-1],
                    'precision': patient_metrics[pid]['precision'][-1],
                    'recall': patient_metrics[pid]['recall'][-1],
                    'hd95': patient_metrics[pid]['hd95'][-1],
                    'assd': patient_metrics[pid]['assd'][-1],
                    'centroid_dist': patient_metrics[pid]['centroid_dist'][-1],
                }
                if compute_cldice:
                    case_entry['cldice'] = patient_metrics[pid]['cldice'][-1]
                
                case_results.append(case_entry)
            
            # Collect data for saving predictions (deferred to separate tqdm loop)
            if (save_predictions or save_masks) and output_dir is not None:
                for i in range(len(patient_ids)):
                    pid = patient_ids[i]
                    meta = batch['metadata'][i] if 'metadata' in batch else {}
                    label_path = meta.get('label_path', 'unknown')
                    slice_val = slice_indices[i].item() if isinstance(slice_indices[i], torch.Tensor) else slice_indices[i]
                    filename_stem = Path(label_path).stem if label_path != 'unknown' else f"{pid}_slice{slice_val:03d}"
                    
                    save_queue.append({
                        'img': images[i, 0].cpu().numpy(),
                        'pred': preds_binary[i, 0].cpu().numpy(),
                        'gt': labels[i, 0].cpu().numpy(),
                        'pid': pid,
                        'slice_idx': slice_val,
                        'dice': patient_metrics[pid]['dice'][-1],
                        'filename_stem': filename_stem,
                    })
    
    # ── Save predictions with separate tqdm ──
    if (save_predictions or save_masks) and output_dir is not None and save_queue:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        mask_dir = Path(output_dir) / 'predictions_mask'
        mask_dir.mkdir(parents=True, exist_ok=True)
        
        if save_predictions:
            qual_dir = Path(output_dir) / 'predictions'
            qual_dir.mkdir(parents=True, exist_ok=True)
        
        for item in tqdm(save_queue, desc="Saving masks/predictions"):
            pid = item['pid']
            slice_idx = item['slice_idx']
            single_dice = item['dice']
            pred_mask = item['pred']
            gt_i = item['gt']
            img_i = item['img']
            
            suffix = "_low" if single_dice < 0.7 else ""
            base_name = f"{item['filename_stem']}_dice{single_dice:.4f}{suffix}"
            
            # ── 1) Save raw mask to predictions_mask/ ──
            pred_mask_uint8 = (pred_mask * 255).astype(np.uint8)
            Image.fromarray(pred_mask_uint8).save(mask_dir / f"{base_name}.png")
            
            # ── 2) Save 3-panel qualitative plot to predictions/ ──
            if save_predictions:
                img_min, img_max = img_i.min(), img_i.max()
                if img_max - img_min > 1e-8:
                    img_display = (img_i - img_min) / (img_max - img_min)
                else:
                    img_display = np.zeros_like(img_i)
                
                H, W = img_display.shape
                
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                
                # Panel 1: Original image
                axes[0].imshow(img_display, cmap='gray', vmin=0, vmax=1)
                axes[0].set_title('Original', fontsize=14)
                axes[0].axis('off')
                
                # Panel 2: Prediction overlay on original
                axes[1].imshow(img_display, cmap='gray', vmin=0, vmax=1)
                tp_mask = (pred_mask > 0.5) & (gt_i > 0.5)
                fp_mask = (pred_mask > 0.5) & (gt_i < 0.5)
                fn_mask = (pred_mask < 0.5) & (gt_i > 0.5)
                
                if tp_mask.any():
                    overlay_tp = np.zeros((H, W, 4), dtype=np.float32)
                    overlay_tp[:, :, 0] = 1.0
                    overlay_tp[:, :, 1] = 1.0
                    overlay_tp[:, :, 2] = 0.0
                    overlay_tp[:, :, 3] = tp_mask.astype(np.float32) * 0.5
                    axes[1].imshow(overlay_tp)
                
                if fp_mask.any():
                    overlay_fp = np.zeros((H, W, 4), dtype=np.float32)
                    overlay_fp[:, :, 0] = 1.0
                    overlay_fp[:, :, 1] = 0.4
                    overlay_fp[:, :, 2] = 0.6
                    overlay_fp[:, :, 3] = fp_mask.astype(np.float32) * 0.6
                    axes[1].imshow(overlay_fp)
                
                if fn_mask.any():
                    overlay_fn = np.zeros((H, W, 4), dtype=np.float32)
                    overlay_fn[:, :, 0] = 0.0
                    overlay_fn[:, :, 1] = 1.0
                    overlay_fn[:, :, 2] = 0.0
                    overlay_fn[:, :, 3] = fn_mask.astype(np.float32) * 0.55
                    axes[1].imshow(overlay_fn)
                
                axes[1].set_title(f'Prediction (Dice={single_dice:.3f})', fontsize=14)
                axes[1].axis('off')
                
                # Panel 3: GT overlay on original (yellow)
                axes[2].imshow(img_display, cmap='gray', vmin=0, vmax=1)
                if gt_i.any():
                    overlay_gt = np.zeros((H, W, 4), dtype=np.float32)
                    overlay_gt[:, :, 0] = 1.0
                    overlay_gt[:, :, 1] = 1.0
                    overlay_gt[:, :, 2] = 0.0
                    overlay_gt[:, :, 3] = (gt_i > 0.5).astype(np.float32) * 0.5
                    axes[2].imshow(overlay_gt)
                
                axes[2].set_title('Ground Truth', fontsize=14)
                axes[2].axis('off')
                
                plt.tight_layout()
                plt.savefig(qual_dir / f"{base_name}.png", dpi=120, bbox_inches='tight')
                plt.close(fig)
        
        del save_queue  # Free memory
    
    # Guard: empty dataset
    if not all_dice_per_sample:
        print("\n⚠ No predictions were made (empty dataset).")
        return {'num_samples': 0}, {}
    
    N = len(all_dice_per_sample)
    
    # ============================================================================
    # Global Dice (from accumulated TP/FP/FN — no large arrays needed)
    # ============================================================================
    dice_per_class = np.array([
        2 * global_tp[c] / (2 * global_tp[c] + global_fp[c] + global_fn[c])
        if (2 * global_tp[c] + global_fp[c] + global_fn[c]) > 0 else np.nan
        for c in range(2)
    ])
    dice_global_fg = dice_per_class[1]
    
    # ============================================================================
    # Per-case Dice (from accumulated per-sample values)
    # ============================================================================
    dice_per_sample = np.array(all_dice_per_sample)
    dice_mean = dice_per_sample.mean()
    dice_std = dice_per_sample.std()
    
    # ============================================================================
    # IoU per class & mIoU (from accumulated counters)
    # ============================================================================
    ious = {}
    for c in range(2):
        if global_union[c] == 0:
            ious[c] = float('nan')
        else:
            ious[c] = float(global_inter[c] / global_union[c])
    
    miou = np.nanmean(list(ious.values()))
    iou_fg = ious.get(1, float('nan'))
    voe_fg = 1.0 - iou_fg if not np.isnan(iou_fg) else float('nan')
    
    # ============================================================================
    # Per-case clDice (grand mean of patient averages)
    # ============================================================================
    all_cldice_per_patient = []
    for pid, metrics in patient_metrics.items():
        if metrics['cldice']:
            all_cldice_per_patient.extend(metrics['cldice'])
    
    if all_cldice_per_patient:
        cldice_mean = np.mean(all_cldice_per_patient)
        cldice_std = np.std(all_cldice_per_patient)
    else:
        cldice_mean = 0.0
        cldice_std = 0.0
    
    # ============================================================================
    # Per-case Surface Distance Metrics (HD95, ASSD, Centroid Distance)
    # ============================================================================
    all_hd95, all_assd, all_cd = [], [], []
    for pid, metrics in patient_metrics.items():
        all_hd95.extend(metrics['hd95'])
        all_assd.extend(metrics['assd'])
        all_cd.extend(metrics['centroid_dist'])
    
    hd95_mean = float(np.nanmean(all_hd95)) if all_hd95 else 0.0
    hd95_std = float(np.nanstd(all_hd95)) if all_hd95 else 0.0
    assd_mean = float(np.nanmean(all_assd)) if all_assd else 0.0
    assd_std = float(np.nanstd(all_assd)) if all_assd else 0.0
    cd_mean = float(np.nanmean(all_cd)) if all_cd else 0.0
    cd_std = float(np.nanstd(all_cd)) if all_cd else 0.0
    
    # ============================================================================
    # Compile all metrics
    # ============================================================================
    avg_metrics = {
        'loss': total_loss / max(num_batches, 1),
        'dice_global_foreground': float(dice_global_fg) if not np.isnan(dice_global_fg) else None,
        'dice_per_case_mean': float(dice_mean),
        'dice_per_case_std': float(dice_std),
        'dice_per_case_min': float(dice_per_sample.min()),
        'dice_per_case_max': float(dice_per_sample.max()),
        'cldice_per_case_mean': float(cldice_mean),
        'cldice_per_case_std': float(cldice_std),
        'hd95_per_case_mean': hd95_mean,
        'hd95_per_case_std': hd95_std,
        'assd_per_case_mean': assd_mean,
        'assd_per_case_std': assd_std,
        'centroid_dist_per_case_mean': cd_mean,
        'centroid_dist_per_case_std': cd_std,
        'miou': float(miou) if not np.isnan(miou) else None,
        'iou_foreground': float(iou_fg) if not np.isnan(iou_fg) else None,
        'voe_foreground': float(voe_fg) if not np.isnan(voe_fg) else None,
        'iou_per_class': {str(k): (float(v) if not np.isnan(v) else None) for k, v in ious.items()},
        'dice_per_class': [float(x) if not np.isnan(x) else None for x in dice_per_class],
        'num_samples': int(N),
    }
    
    # Detection Success Rate (fg IoU > 0.3 = detected)
    all_iou_per_sample = [cr['iou'] for cr in case_results]
    n_detected = sum(1 for iou in all_iou_per_sample if iou > 0.3)
    detection_success_rate = n_detected / N if N > 0 else 0.0
    avg_metrics['detection_success_rate'] = float(detection_success_rate)
    avg_metrics['detection_count'] = n_detected
    
    # Calculate per-patient average metrics
    patient_avg_metrics = {}
    for pid, metrics in patient_metrics.items():
        entry = {
            'dice': np.mean(metrics['dice']),
            'iou': np.mean(metrics['iou']),
            'voe': np.mean(metrics['voe']),
            'precision': np.mean(metrics['precision']),
            'recall': np.mean(metrics['recall']),
            'num_slices': len(metrics['dice']),
            'hd95': float(np.nanmean(metrics['hd95'])) if metrics['hd95'] else np.nan,
            'assd': float(np.nanmean(metrics['assd'])) if metrics['assd'] else np.nan,
            'centroid_dist': float(np.nanmean(metrics['centroid_dist'])) if metrics['centroid_dist'] else np.nan,
        }
        if metrics['cldice']:
            entry['cldice'] = np.mean(metrics['cldice'])
        patient_avg_metrics[pid] = entry
    
    return avg_metrics, patient_avg_metrics, case_results


# ============================================================================
# Main
# ============================================================================

def main(args):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    if args.output_dir is None:
        tz = pytz.timezone('Asia/Seoul')
        timestamp = datetime.now(tz).strftime('%Y_%m_%d_%H_%M_%S')
        args.output_dir = f'validation_results_{timestamp}'
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("UNet Validation (Slice)")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Fold: {args.fold}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    # ── Load validation dataset (npy slices from dataset_2d_slice) ──
    print("\nLoading validation dataset (npy slices)...")
    target_size = (args.image_size, args.image_size) if args.image_size else None
    
    val_dataset = TMJ_dataset_2D(
        metadata_file=args.metadata_file,
        data_dir=args.data_dir,
        label_dir=args.label_dir,
        split=args.split,
        fold=args.fold,
        val_ratio=args.val_ratio,
        fold_seed=args.seed,
        normalize=args.normalize,
        target_size=target_size,
        splits_json_path=args.splits_json if args.splits_json else None,
    )
    
    print(f"Validation samples (fold {args.fold}, split='{args.split}'): {len(val_dataset)}")
    
    if len(val_dataset) == 0:
        print("\n⚠ No samples found! Check --fold, --split, and --splits_json arguments.")
        print("  splits_final.json typically has 'train' and 'val' keys (no 'test').")
        print("  Try: --split val")
        return
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn_2d,
    )
    
    # ── Load checkpoint to get model configuration ──
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    ckpt_args = checkpoint.get('args', {}) if isinstance(checkpoint.get('args'), dict) else {}
    if ckpt_args and 'model' in ckpt_args:
        model_name = ckpt_args['model']
        n_channels = ckpt_args.get('n_channels', args.n_channels)
        print(f"Model type from checkpoint: {model_name}")
    else:
        model_name = args.model
        n_channels = args.n_channels
        print(f"Model type from argument: {model_name}")

    if '--model' in sys.argv:
        model_name = args.model
        print(f"Model type overridden by --model: {model_name}")

    main_slice_weight = ckpt_args.get('main_slice_weight', 1.0)
    learnable_query_ckpt = ckpt_args.get('learnable_query_ckpt', '')
    
    # ── Load MedDINOv3 ──
    dinov3_model = None
    print("Loading MedDINOv3...")
    try:
        meddinov3_path = Path("/path/to/MedDINOv3")
        if str(meddinov3_path) not in sys.path:
            sys.path.insert(0, str(meddinov3_path))
        from dinov3.models.vision_transformer import vit_base

        dinov3_model = vit_base(
            patch_size=16, img_size=518, init_values=1.0,
            block_chunks=0, num_register_tokens=0,
            interpolate_antialias=False, interpolate_offset=0.1,
        )

        dinov3_checkpoint = "/path/to/MedDINOv3/checkpoint/model.pth"
        if os.path.exists(dinov3_checkpoint):
            ckpt = torch.load(dinov3_checkpoint, map_location='cpu')
            state_dict = ckpt.get('teacher', ckpt.get('model', ckpt))
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k
                if new_key.startswith('module.'):
                    new_key = new_key[7:]
                if new_key.startswith('backbone.'):
                    new_key = new_key[9:]
                new_state_dict[new_key] = v
            dinov3_model.load_state_dict(new_state_dict, strict=False)
            n_matched = len(set(dinov3_model.state_dict().keys()) & set(new_state_dict.keys()))
            print(f"✓ MedDINOv3 loaded (matched {n_matched}/{len(dinov3_model.state_dict())} keys)")

        dinov3_model.eval()
        for p in dinov3_model.parameters():
            p.requires_grad = False
        dinov3_model = dinov3_model.to(device)
    except Exception as e:
        print(f"⚠ Failed to load MedDINOv3: {e}")
        dinov3_model = None
    
    # ── Build model ──
    print("\nBuilding model...")
    model = get_model(
        model_name,
        n_channels=n_channels,
        n_classes=args.n_classes,
        bilinear=args.bilinear,
        dinov3_model=dinov3_model,
        dinov3_feature_dim=768,
        dinov3_input_size=518,
        main_slice_weight=main_slice_weight,
        learnable_query_ckpt=learnable_query_ckpt,
    )
    
    # Disable PointRend for inference (coarse → iterative refinement handled by model.eval())
    if hasattr(model, 'enable_point_rend'):
        if args.warm_up:
            model.enable_point_rend = False
            print('Running in warm_up mode: PointRend is disabled.')
        else:
            model.enable_point_rend = True
            print('Running in standard mode: PointRend is enabled.')
    
    model = model.to(device)
    
    # ── Load weights (flexible, shape-aware) ──
    if 'model_state_dict' in checkpoint:
        state_dict_key = 'model_state_dict'
    elif 'model' in checkpoint and isinstance(checkpoint['model'], dict):
        state_dict_key = 'model'
    else:
        # Assume checkpoint IS the state dict
        state_dict_key = None
    
    if state_dict_key:
        pretrained_dict = checkpoint[state_dict_key]
    else:
        pretrained_dict = checkpoint
    
    # Filter by shape
    model_dict = model.state_dict()
    filtered_dict = {
        k: v for k, v in pretrained_dict.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }
    n_loaded = len(filtered_dict)
    n_total = len(model_dict)
    n_skipped = len(pretrained_dict) - n_loaded
    
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)
    print(f"✓ Loaded {n_loaded}/{n_total} parameters (skipped {n_skipped} shape-mismatched)")
    
    if 'epoch' in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")
    if 'metrics' in checkpoint:
        print(f"Checkpoint metrics: {checkpoint['metrics']}")
    
    # ── Validate ──
    print("\n" + "=" * 80)
    print("Running validation...")
    print("=" * 80)
    
    # Set args.model so validate() knows model type
    args.model = model_name
    
    avg_metrics, patient_metrics, case_results = validate(
        model, val_loader, device, args,
        save_predictions=args.save_predictions,
        save_masks=args.save_masks,
        output_dir=output_dir,
    )
    
    # ── Print results ──
    print("\n" + "=" * 80)
    print("Validation Results")
    print("=" * 80)
    print(f"Samples:       {avg_metrics['num_samples']}")
    print(f"Average Loss:  {avg_metrics['loss']:.4f}")
    print("\n--- Dice (for paper report) ---")
    if avg_metrics['dice_global_foreground'] is not None:
        print(f"  Global Dice (foreground):             {avg_metrics['dice_global_foreground']:.4f}")
    print(f"  Per-case mean Dice (±std):            {avg_metrics['dice_per_case_mean']:.4f} ± {avg_metrics['dice_per_case_std']:.4f}")
    print(f"  Per-case Dice (min/max):              {avg_metrics['dice_per_case_min']:.4f} / {avg_metrics['dice_per_case_max']:.4f}")
    print("\n--- clDice (Centerline Dice) ---")
    print(f"  Per-case mean clDice (±std):          {avg_metrics['cldice_per_case_mean']:.4f} ± {avg_metrics['cldice_per_case_std']:.4f}")
    print("\n--- Surface Distance Metrics ---")
    print(f"  HD95 (±std):                          {avg_metrics['hd95_per_case_mean']:.2f} ± {avg_metrics['hd95_per_case_std']:.2f}")
    print(f"  ASSD (±std):                          {avg_metrics['assd_per_case_mean']:.2f} ± {avg_metrics['assd_per_case_std']:.2f}")
    print(f"  Centroid Distance (±std):             {avg_metrics['centroid_dist_per_case_mean']:.2f} ± {avg_metrics['centroid_dist_per_case_std']:.2f}")
    print("\n--- Detection ---")
    print(f"  Detection Success Rate (IoU>0.3):     {avg_metrics['detection_success_rate']:.4f} ({avg_metrics['detection_count']}/{avg_metrics['num_samples']})")
    print("---")
    if avg_metrics['miou'] is not None:
        print(f"mIoU:          {avg_metrics['miou']:.4f}")
    if avg_metrics['iou_foreground'] is not None:
        print(f"IoU (fg):      {avg_metrics['iou_foreground']:.4f}")
    if avg_metrics['voe_foreground'] is not None:
        print(f"VOE (fg):      {avg_metrics['voe_foreground']:.4f}")
    print(f"IoU per class: {avg_metrics['iou_per_class']}")
    print(f"Dice per class: {avg_metrics['dice_per_class']}")
    print("=" * 80)
    
    # ── Save metrics to JSON ──
    results_json = output_dir / 'validation_metrics.json'
    with open(results_json, 'w') as f:
        json.dump(avg_metrics, f, indent=2)
    print(f"\nOverall metrics saved to: {results_json}")
    
    # ── Save as CSV ──
    results_csv = output_dir / 'validation_metrics.csv'
    overall_df = pd.DataFrame([{
        'num_samples': avg_metrics['num_samples'],
        'loss': avg_metrics['loss'],
        'dice_global_foreground': avg_metrics['dice_global_foreground'],
        'dice_per_case_mean': avg_metrics['dice_per_case_mean'],
        'dice_per_case_std': avg_metrics['dice_per_case_std'],
        'cldice_per_case_mean': avg_metrics['cldice_per_case_mean'],
        'cldice_per_case_std': avg_metrics['cldice_per_case_std'],
        'hd95_per_case_mean': avg_metrics['hd95_per_case_mean'],
        'hd95_per_case_std': avg_metrics['hd95_per_case_std'],
        'assd_per_case_mean': avg_metrics['assd_per_case_mean'],
        'assd_per_case_std': avg_metrics['assd_per_case_std'],
        'centroid_dist_per_case_mean': avg_metrics['centroid_dist_per_case_mean'],
        'centroid_dist_per_case_std': avg_metrics['centroid_dist_per_case_std'],
        'detection_success_rate': avg_metrics['detection_success_rate'],
        'detection_count': avg_metrics['detection_count'],
        'miou': avg_metrics['miou'],
        'iou_foreground': avg_metrics['iou_foreground'],
        'voe_foreground': avg_metrics['voe_foreground'],
    }])
    overall_df.to_csv(results_csv, index=False)
    print(f"Overall metrics (simplified) saved to: {results_csv}")
    
    # ── Per-patient metrics ──
    patient_csv = output_dir / 'patient_metrics.csv'
    patient_df = pd.DataFrame.from_dict(patient_metrics, orient='index')
    patient_df.index.name = 'patient_id'
    patient_df = patient_df.sort_values('dice', ascending=False)
    patient_df.to_csv(patient_csv)
    print(f"Per-patient metrics saved to: {patient_csv}")

    # ── Per-case metrics (for t-test) ──
    case_csv = output_dir / 'validation_metrics_case_level.csv'
    case_df = pd.DataFrame(case_results)
    # Reorder columns: filename first
    cols = ['filename'] + [c for c in case_df.columns if c != 'filename']
    case_df = case_df[cols]
    case_df.to_csv(case_csv, index=False)
    print(f"Per-case metrics saved to: {case_csv}")
    
    # ── Print top and bottom patients ──
    print("\n" + "=" * 80)
    print("Top 5 patients by Dice score:")
    print("=" * 80)
    for i, (pid, metrics) in enumerate(patient_df.head(5).iterrows(), 1):
        cldice_str = f", clDice={metrics.get('cldice', 0):.4f}" if 'cldice' in metrics else ""
        print(f"{i}. Patient {pid}: Dice={metrics['dice']:.4f}, IoU={metrics['iou']:.4f}{cldice_str} ({int(metrics['num_slices'])} slices)")
    
    print("\n" + "=" * 80)
    print("Bottom 5 patients by Dice score:")
    print("=" * 80)
    for i, (pid, metrics) in enumerate(patient_df.tail(5).iterrows(), 1):
        cldice_str = f", clDice={metrics.get('cldice', 0):.4f}" if 'cldice' in metrics else ""
        print(f"{i}. Patient {pid}: Dice={metrics['dice']:.4f}, IoU={metrics['iou']:.4f}{cldice_str} ({int(metrics['num_slices'])} slices)")
    
    print("\n" + "=" * 80)
    print("Validation completed!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='UNet Validation for TMJ 2D Segmentation (npy slice data)')
    
    # Dataset arguments (npy slice based)
    parser.add_argument('--metadata_file', type=str,
                        default='/path/to/data.csv',
                        help='Path to metadata CSV file')
    parser.add_argument('--data_dir', type=str,
                        default='/path/to/data_converted/nifti',
                        help='Path to pre-converted npy slice directory')
    parser.add_argument('--label_dir', type=str,
                        default='/path/to/data/labels',
                        help='Path to label directory')
    parser.add_argument('--fold', type=int, required=True,
                        help='Fold number for validation')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                        help='Dataset split to validate on (default: val). '
                             'Note: splits_final.json has train/val only, no test.')
    parser.add_argument('--splits_json', type=str, default='/path/to/splits_final.json',
                        help='Path to splits_final.json (optional, for nnUNet-style splits)')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='(Dataset internal) train/val split ratio')
    parser.add_argument('--normalize', type=str, default='minmax', choices=['minmax', 'zscore', 'none'],
                        help='Normalization method')
    parser.add_argument('--image_size', type=int, default=512,
                        help='Resize image/label to (image_size, image_size). Set 0 to disable.')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='unet_all_mol_learnable_query',
                        help='Model type (only unet_all_mol_learnable_query is supported)')
    parser.add_argument('--n_channels', type=int, default=1,
                        help='Number of input channels')
    parser.add_argument('--n_classes', type=int, default=1,
                        help='Number of output classes')
    parser.add_argument('--bilinear', action='store_true',
                        help='Use bilinear upsampling instead of transposed convolution')
    parser.add_argument('--warm_up', action='store_true',
                        help='Evaluate warmup phase logic (PointRend disabled during inference)')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    
    # Validation arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Validation batch size')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save prediction masks and overlay images')
    parser.add_argument('--save_masks', action='store_true',
                        help='Save ONLY prediction masks without qualitative overlays')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save results (default: auto-generated)')
    parser.add_argument('--cldice', action='store_true', default=True,
                        help='Compute clDice metric (enabled by default)')
    
    # System arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    args = parser.parse_args()
    
    main(args)
