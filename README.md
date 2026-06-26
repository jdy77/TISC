# [MICCAI 2026] TISC

Official repository for [MICCAI 2026] Anatomically Consistent TMJ Disc Segmentation via Semantic Anchoring and Clinical Priors (TISC).

---

## Overview

This repository implements the TISC framework, designed for anatomically consistent temporomandibular joint (TMJ) disc segmentation using semantic anchoring and clinical priors. The training pipeline consists of two main stages:
1. **Learnable Query Module Training**: Trains the `CrossAttentionQueryExtractor` to focus on the TMJ disc region.
2. **Full Segmentation Training**: Trains the main UNet segmentation model integrated with the pre-trained and frozen query module.

For evaluation and inference, a separate validation script computes standard segmentation metrics (Dice, clDice, HD95, ASSD, CD) and saves qualitative visualizations.

---

## Workflow

### 1. Learnable Query Module Training

In the first stage, the `CrossAttentionQueryExtractor` is trained standalone so that its cross-attention maps align with the target masks.

```bash
python train_slice_module_learnable_query.py \
    --fold 0 \
    --epochs 100 \
    --batch_size 8 \
    --rotate \
    --elastic \
    --zoom \
    --save_attn_epoch 1 \
    --save_attn_max 5 \
    --save_epochly
```

- **Arguments**:
  - `--fold`: Index of the cross-validation fold (0-4).
  - `--epochs`: Number of training epochs.
  - `--rotate`, `--elastic`, `--zoom`: Enable data augmentations.
  - `--save_attn_epoch`: Frequency of attention map visual savings.
- **Output**: Checkpoints are saved under `./results/module/learnable_query_aug/fold{fold}/checkpoints/`.

### 2. Full Segmentation Model Training

Once the query module is trained, the full segmentation model is trained. The query module's checkpoint is loaded and frozen, and its output is used to dynamically adapt DINO features injected into the UNet bottleneck.

```bash
python train_slice.py \
    --model unet_all_mol_learnable_query \
    --fold 0 \
    --splits_json /path/to/splits_final.json \
    --epochs 100 \
    --warm_up 10 \
    --point_loss_weight 20 \
    --coarse_weight 0.8 \
    --exp_name "TISC_fold0" \
    --log_loss \
    --cldice \
    --main_slice_weight 2.0 \
    --elastic \
    --rotation \
    --zoom \
    --save_epochly \
    --learnable_query_ckpt ./results/module/learnable_query_aug/fold0/checkpoints/best_query_module.pth
```

- **Arguments**:
  - `--model`: Main model architecture (`unet_all_mol_learnable_query`).
  - `--learnable_query_ckpt`: Path to the pre-trained query module checkpoint from Stage 1.
  - `--cldice`: Evaluates the centerline Dice metric during validation.
  - `--warm_up`: Warmup epochs before starting PointRend boundary refinement.

### 3. Inference and Evaluation

To validate a trained checkpoint, compute evaluation metrics, and export prediction overlays:

```bash
python validate_slice.py \
    --fold 0 \
    --checkpoint ./results/UNet/unet_all_mol_learnable_query/TISC_fold0/checkpoints/best_epoch037_0.8012.pth \
    --output_dir validation_results/fold0 \
    --normalize minmax \
    --num_workers 4 \
    --seed 42 \
    --model unet_all_mol_learnable_query \
    --save_predictions
```

- **Arguments**:
  - `--checkpoint`: Path to the trained full model checkpoint (required).
  - `--save_predictions`: Save predicted overlay mask comparisons (Original / Prediction / GT).
- **Output**:
  - `validation_metrics.csv` / `validation_metrics.json`: Overall summary statistics.
  - `patient_metrics.csv`: Patient-level average metric breakdowns.
  - `predictions/`: Visual comparisons of the prediction masks.

---

## File Structure

```
.
├── train_slice_module_learnable_query.py # Stage 1: Learnable Query training
├── train_slice.py                        # Stage 2: Full model training
├── validate_slice.py                     # Stage 3: Inference and validation
├── dataset_2d_slice.py                   # 2D slice Dataset and custom augmentations
├── convert_nifti_to_npy_slices.py        # Utility to preprocess NIfTI volumes into slices
├── models/
│   ├── model_unet_all_mol_learnable_query.py # Full UNet model architecture
│   └── module_learnable_query.py             # Learnable Query extractor definition
├── run_train.sh                          # Example training command script
└── run_validation.sh                     # Example validation command script
```
