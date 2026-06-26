"""
TMJ 2D Segmentation Dataset (pre-converted slices)

Uses pre-converted 2D slices (.npy) from NIfTI volumes, returning each slice-label pair.

Directory structure:

    Original NIfTI:
        /path/to/data/nifti

    Converted slices (.npy):
        /path/to/data_converted/nifti/<nifti_basename>/<zzz>.npy

Where <nifti_basename> is NIfTI name without extension, and <zzz> is 3-digit zero-padded z-index (e.g. 014.npy).

Label PNG:
    /path/to/data/labels/<nifti_basename>_<zzz>.png

Example:
    302562_FL_A_PD_SAG_OPEN.nii.gz
Matches:
    302562_FL_A_PD_SAG_OPEN/014.npy  ← z=14 slice
    302562_FL_A_PD_SAG_OPEN_014.png  ← label for z=14
"""

import json
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Union
import numpy as np
import pandas as pd
import nibabel as nib
from PIL import Image
import torch
from torch.utils.data import Dataset
import cv2
from scipy.ndimage import gaussian_filter, map_coordinates


# ──────────────────────────────────────────────────────────────
#  Augmentation transforms (dict-based: image [H,W] or [3,H,W], label [H,W])
# ──────────────────────────────────────────────────────────────

class ElasticTransform:
    """
    Elastic deformation for 2D medical image segmentation.
    
    Applies the same elastic deformation grid to image, label, and dino_slices (if present).
    - image: bilinear interpolation
    - label: nearest neighbor interpolation
    - dino_slices [3,H,W]: bilinear interpolation

    Args:
        alpha: Distortion scale (default 80.0). Larger values mean stronger deformation.
        sigma: Gaussian smoothing standard deviation (default 8.0).
        p: Probability of applying transform (default 0.5).
    """
    def __init__(self, alpha: float = 80.0, sigma: float = 8.0, p: float = 0.5):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def __call__(self, data: dict) -> dict:
        if np.random.rand() > self.p:
            return data

        image = data['image']   # [H, W] or [3, H, W]
        label = data['label']   # [H, W]

        # 2D spatial shape
        if image.ndim == 3:
            H, W = image.shape[1], image.shape[2]
        else:
            H, W = image.shape

        # Random displacement field + Gaussian smoothing
        dx = gaussian_filter(np.random.randn(H, W) * self.alpha, self.sigma)
        dy = gaussian_filter(np.random.randn(H, W) * self.alpha, self.sigma)

        y_grid, x_grid = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        indices_y = np.clip(y_grid + dy, 0, H - 1)
        indices_x = np.clip(x_grid + dx, 0, W - 1)

        # Image deformation
        if image.ndim == 3:
            # 2.5D: Apply same grid to each channel
            new_image = np.empty_like(image)
            for c in range(image.shape[0]):
                new_image[c] = map_coordinates(
                    image[c], [indices_y, indices_x], order=1, mode='reflect'
                ).reshape(H, W)
            data['image'] = new_image
        else:
            data['image'] = map_coordinates(
                image, [indices_y, indices_x], order=1, mode='reflect'
            ).reshape(H, W)

        # Label deformation (nearest neighbor to keep binary mask)
        data['label'] = map_coordinates(
            label.astype(np.float32), [indices_y, indices_x], order=0, mode='reflect'
        ).reshape(H, W).astype(label.dtype)

        # Deform dino_slices [3, H, W] — Apply same grid (bilinear)
        if 'dino_slices' in data:
            dino = data['dino_slices']
            new_dino = np.empty_like(dino)
            for c in range(dino.shape[0]):
                new_dino[c] = map_coordinates(
                    dino[c], [indices_y, indices_x], order=1, mode='reflect'
                ).reshape(H, W)
            data['dino_slices'] = new_dino

        return data


class RandomRotation:
    """
    Random rotation augmentation for 2D medical image segmentation.

    Applies the same rotation to image, label, and dino_slices.
    - image: bilinear interpolation
    - label: nearest neighbor interpolation
    - dino_slices [3,H,W]: bilinear interpolation

    Args:
        degree_range: Range (min_deg, max_deg) to sample uniform random angle.
        p: Probability of applying transform.
    """
    def __init__(self, degree_range: tuple = (-15, 15), p: float = 0.5):
        self.degree_range = degree_range
        self.p = p

    def __call__(self, data: dict) -> dict:
        if np.random.rand() > self.p:
            return data

        image = data['image']
        label = data['label']
        angle = np.random.uniform(self.degree_range[0], self.degree_range[1])

        if image.ndim == 3:
            H, W = image.shape[1], image.shape[2]
        else:
            H, W = image.shape

        center = (W / 2.0, H / 2.0)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        if image.ndim == 3:
            new_image = np.empty_like(image)
            for c in range(image.shape[0]):
                new_image[c] = cv2.warpAffine(
                    image[c], M, (W, H),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
                )
            data['image'] = new_image
        else:
            data['image'] = cv2.warpAffine(
                image, M, (W, H),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
            )

        rotated_label = cv2.warpAffine(
            label.astype(np.float32), M, (W, H),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101,
        ).astype(label.dtype)
        data['label'] = rotated_label

        # Rotate dino_slices [3, H, W] using same rotation matrix
        if 'dino_slices' in data:
            dino = data['dino_slices']
            new_dino = np.empty_like(dino)
            for c in range(dino.shape[0]):
                new_dino[c] = cv2.warpAffine(
                    dino[c], M, (W, H),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
                )
            data['dino_slices'] = new_dino

        return data


class RandomZoom:
    """
    Random zoom (scale) augmentation for 2D medical image segmentation.

    Zooms from center by (1+min_frac) to (1+max_frac), then crops/pads back to (H, W).
    - image: bilinear interpolation
    - label: nearest neighbor interpolation
    - dino_slices [3,H,W]: bilinear interpolation

    Args:
        scale_range: (min_frac, max_frac). e.g., (0.0, 0.1) -> 1.0x ~ 1.1x zoom.
        p: Probability of applying transform.
    """
    def __init__(self, scale_range: tuple = (0.0, 0.1), p: float = 0.5):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, data: dict) -> dict:
        if np.random.rand() > self.p:
            return data

        image = data['image']
        label = data['label']
        scale = 1.0 + np.random.uniform(self.scale_range[0], self.scale_range[1])

        if image.ndim == 3:
            H, W = image.shape[1], image.shape[2]
        else:
            H, W = image.shape

        center = (W / 2.0, H / 2.0)
        M = cv2.getRotationMatrix2D(center, 0, scale)

        if image.ndim == 3:
            new_image = np.empty_like(image)
            for c in range(image.shape[0]):
                new_image[c] = cv2.warpAffine(
                    image[c], M, (W, H),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
                )
            data['image'] = new_image
        else:
            data['image'] = cv2.warpAffine(
                image, M, (W, H),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
            )

        zoomed_label = cv2.warpAffine(
            label.astype(np.float32), M, (W, H),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101,
        ).astype(label.dtype)
        data['label'] = zoomed_label

        # Zoom dino_slices [3, H, W] using same scale matrix
        if 'dino_slices' in data:
            dino = data['dino_slices']
            new_dino = np.empty_like(dino)
            for c in range(dino.shape[0]):
                new_dino[c] = cv2.warpAffine(
                    dino[c], M, (W, H),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
                )
            data['dino_slices'] = new_dino

        return data


class ComposeTransforms:
    """Apply multiple dict-based transforms sequentially."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data: dict) -> dict:
        for t in self.transforms:
            data = t(data)
        return data


class TMJ_dataset_2D(Dataset):
    """
    TMJ 2D Segmentation Dataset.
    Treats each slice in a NIfTI volume as an individual sample.
    Only uses slices that have a corresponding label PNG.

    Args:
        metadata_file: Path to CSV file containing Patient ID, fold, etc.
        data_dir: Directory containing NIfTI converted slices.
        label_dir: Directory containing label PNG files.
        split: 'train', 'val', or 'test' (None for all).
        fold: If specified, this fold is test, and remainder is 80% train / 20% val (subject-level).
              If None, uses 'set' column in CSV.
        val_ratio: Validation ratio from non-test subjects (default 0.2).
        fold_seed: Seed for train/val split (default 42).
        transform: Optional data augmentation.
        normalize: Normalization method ('minmax', 'zscore', or None).
        target_size: (H, W) target size to resize. None keeps original size.
    """
    
    def __init__(
        self,
        metadata_file: str = "/path/to/data.csv",
        data_dir: str = "/path/to/data_converted/nifti",
        label_dir: str = "/path/to/data/labels",
        split: Optional[str] = None,
        fold: Optional[int] = None,
        val_ratio: float = 0.2,
        fold_seed: int = 42,
        transform=None,
        normalize: str = 'minmax',
        target_size: Optional[Tuple[int, int]] = None,
        splits_json_path: Optional[str] = None,
        use_2_5d: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform
        self.split = split
        self.normalize = normalize
        self.fold = fold
        self.val_ratio = val_ratio
        self.fold_seed = fold_seed
        self.target_size = target_size  # (H, W)
        self.use_2_5d = use_2_5d  # 2.5D: [prev, curr, next] slices
        self.splits_json_path = splits_json_path
        # Do not use NIfTI cache since we load pre-converted .npy slices
        
        # Load metadata
        self.metadata = pd.read_csv(metadata_file)
        
        # If splits_json is used: filter train/val by nnUNet slice IDs
        self._split_ids: Optional[set] = None
        if splits_json_path and split is not None and fold is not None:
            with open(splits_json_path) as f:
                splits_data = json.load(f)
            if fold < len(splits_data) and split in splits_data[fold]:
                self._split_ids = set(splits_data[fold][split])
            else:
                self._split_ids = set()
        elif fold is not None and 'fold' in self.metadata.columns:
            # CSV fold based: split fold into train/val/test
            self._patient_ids_in_split = self._build_patient_ids_by_fold(split, fold, val_ratio, fold_seed)
        else:
            # CSV set partition-based subject split
            self._subject_to_set = self._build_subject_to_set()
            if split is not None:
                self._patient_ids_in_split = {
                    pid for pid, s in self._subject_to_set.items() if s == split
                }
            else:
                self._patient_ids_in_split = set(self.metadata['Patient ID'].astype(str).unique())
        
        # Build 2D slice sample list (only slices with labels)
        # MOL lookup mapping
        self._patient_mol = self._build_patient_mol_lookup()
        self.samples = self._build_2d_dataset()
        
        print(f"TMJ_dataset_2D initialized:")
        if self._split_ids is not None:
            print(f"  - Split: {split} (splits_final.json fold {fold}, {len(self._split_ids)} IDs)")
        elif fold is not None:
            print(f"  - Split: {split if split else 'all'} (fold-based: fold {fold}=test, rest 80% train / 20% val)")
        else:
            print(f"  - Split: {split if split else 'all'} (subject-level: same Patient ID → same set)")
        print(f"  - Total 2D samples: {len(self.samples)}")
        print(f"  - Normalization: {normalize}")
        print(f"  - Target size (resize): {target_size if target_size else 'None (keep original)'}")
        print(f"  - Data dir: {self.data_dir}")
        print(f"  - Label dir: {self.label_dir}")
    
    def _build_patient_ids_by_fold(
        self,
        split: Optional[str],
        fold: int,
        val_ratio: float,
        seed: int
    ) -> set:
        """
        Fold-based: fold==test, rest are split into 80% train / 20% val (subject-level).
        """
        id_col = 'Patient ID' if 'Patient ID' in self.metadata.columns else 'subject_id'
        # Fold for each subject (based on first row)
        subject_to_fold: Dict[str, int] = {}
        for _, row in self.metadata.iterrows():
            pid = str(row[id_col])
            if pid not in subject_to_fold:
                f = row['fold']
                if pd.isna(f):
                    subject_to_fold[pid] = 0
                else:
                    try:
                        subject_to_fold[pid] = int(float(f))
                    except (ValueError, TypeError):
                        subject_to_fold[pid] = 0
        test_ids = {pid for pid, f in subject_to_fold.items() if f == fold}
        remaining_ids = [pid for pid, f in subject_to_fold.items() if f != fold]
        rng = np.random.default_rng(seed)
        rng.shuffle(remaining_ids)
        n_val = max(1, int(len(remaining_ids) * val_ratio))
        val_ids = set(remaining_ids[:n_val])
        train_ids = set(remaining_ids[n_val:])
        if split is None:
            return set(subject_to_fold.keys())
        if split == 'train':
            return train_ids
        if split == 'val':
            return val_ids
        if split == 'test':
            return test_ids
        return set()
    
    def _build_patient_mol_lookup(self) -> Dict[str, int]:
        """
        Maps patient_id to MOL (0 or 1) from CSV. Defaults to 0.
        """
        lookup = {}
        if 'MOL' not in self.metadata.columns:
            return lookup
        id_col = 'Patient ID' if 'Patient ID' in self.metadata.columns else 'subject_id'
        for _, row in self.metadata.iterrows():
            pid = str(row[id_col])
            if pid not in lookup:
                mol_val = row.get('MOL', 0)
                if pd.isna(mol_val):
                    mol_val = 0
                try:
                    mol_val = int(float(mol_val))
                except (ValueError, TypeError):
                    mol_val = 0
                # Clamp to 0 or 1
                lookup[pid] = 1 if mol_val >= 1 else 0
        return lookup
    
    def _build_subject_to_set(self) -> Dict[str, str]:
        """
        Assigns one set partition ('train', 'val', 'test') per subject_id.
        """
        subject_to_set = {}
        id_col = 'Patient ID' if 'Patient ID' in self.metadata.columns else 'subject_id'
        set_col = 'set'
        if set_col not in self.metadata.columns:
            return subject_to_set
        for _, row in self.metadata.iterrows():
            pid = str(row[id_col])
            if pid not in subject_to_set:
                subject_to_set[pid] = str(row[set_col]).strip().lower()
            elif subject_to_set[pid] != str(row[set_col]).strip().lower():
                import warnings
                warnings.warn(
                    f"Patient ID '{pid}' has inconsistent 'set' in CSV; using first value '{subject_to_set[pid]}'."
                )
        return subject_to_set
    
    def _parse_label_filename(self, label_path: Path) -> Optional[Tuple[str, int]]:
        """
        Extracts (nifti_basename, slice_idx) from label filename.
        e.g., 302562_FL_A_PD_SAG_OPEN_014.png -> ('302562_FL_A_PD_SAG_OPEN', 14)
        """
        stem = label_path.stem  # Exclude extension
        parts = stem.split('_')
        if len(parts) < 2:
            return None
        try:
            slice_idx = int(parts[-1])
        except (ValueError, TypeError):
            return None
        basename = '_'.join(parts[:-1])
        return (basename, slice_idx)

    def _build_2d_dataset(self) -> List[Dict]:
        """
        Iterate over all label PNG files and add valid slice pairs to dataset.
        
        Returns:
            List of dict containing:
                - slice_npy_path: Path to slice .npy file
                - slice_idx: Slice index (0-based, same as 3-digit zero-padding)
                - label_path: Path to label PNG file
                - side: 'right' or 'left'
                - patient_id: Patient ID
                - scan_type: OPEN or CLOSE
        """
        samples: List[Dict] = []
        skipped_no_npy = 0
        skipped_wrong_split = 0
        skipped_not_in_metadata = 0
        skipped_not_in_splits = 0

        # Scan label directory
        all_labels = sorted(self.label_dir.glob("*.png"))

        use_splits_json = getattr(self, "_split_ids", None) is not None

        for label_path in all_labels:
            parsed = self._parse_label_filename(label_path)
            if parsed is None:
                continue
            nifti_basename, slice_idx = parsed
            patient_id = nifti_basename.split('_')[0]

            if use_splits_json:
                # splits_final.json format: nnUNet ID = "basename_slice006"
                nnunet_id = f"{nifti_basename}_slice{slice_idx:03d}"
                if nnunet_id not in self._split_ids:
                    skipped_not_in_splits += 1
                    continue
            else:
                # Skip patients not in metadata
                if patient_id not in self.metadata['Patient ID'].astype(str).values:
                    skipped_not_in_metadata += 1
                    continue
                # Skip patients not in current split
                if not self._is_valid_patient(patient_id):
                    skipped_wrong_split += 1
                    continue

            slice_npy_path = self.data_dir / nifti_basename / f"{slice_idx:03d}.npy"
            if not slice_npy_path.exists():
                skipped_no_npy += 1
                continue

            side = 'right' if slice_idx <= 10 else 'left'
            scan_type = 'OPEN' if 'OPEN' in nifti_basename else 'CLOSE'

            samples.append({
                'slice_npy_path': str(slice_npy_path),
                'slice_idx': slice_idx,
                'label_path': str(label_path),
                'side': side,
                'patient_id': patient_id,
                'scan_type': scan_type,
                'nifti_basename': nifti_basename,
                'mol': self._patient_mol.get(patient_id, 0),
            })

        if use_splits_json and skipped_not_in_splits > 0:
            print(f"  - Skipped (not in splits_final.json this split): {skipped_not_in_splits}")
        if skipped_no_npy > 0:
            print(f"  - Skipped (missing .npy): {skipped_no_npy}")
        if not use_splits_json and (skipped_no_npy > 0 or skipped_wrong_split > 0 or skipped_not_in_metadata > 0):
            print(
                f"  - Skipped: {skipped_not_in_metadata} not in metadata, "
                f"{skipped_wrong_split} other split, {skipped_no_npy} missing .npy "
                f"(total labels in dir: {len(all_labels)})"
            )
        return samples
    
    def _is_valid_patient(self, patient_id: str) -> bool:
        """Check if patient is in the current split."""
        pid = str(patient_id)
        if pid not in self.metadata['Patient ID'].astype(str).values:
            return False
        return pid in self._patient_ids_in_split
    
    def _find_all_label_files(self, nifti_basename: str) -> List[Tuple[Path, int, str]]:
        """
        Find all label PNGs matching NIfTI basename.
        
        Returns:
            List of (label_path, slice_idx, side)
        """
        pattern = f"{nifti_basename}_*.png"
        matching_labels = list(self.label_dir.glob(pattern))
        
        label_info = []
        
        for label_path in matching_labels:
            label_name = label_path.stem
            try:
                slice_idx = int(label_name.split('_')[-1])
                side = 'right' if slice_idx <= 10 else 'left'
                label_info.append((label_path, slice_idx, side))
            except (ValueError, IndexError):
                continue
        
        return label_info
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, any]:
        """
        Load 2D slice data.
        """
        sample = self.samples[idx]
        
        if self.use_2_5d:
            # 2.5D: [prev, curr, next] slices concat
            image_slice = self._load_2_5d_slices(sample)
        else:
            # 2D: current slice only
            image_slice = self._load_npy_slice(sample['slice_npy_path'])
        
        # Always load adjacent slices for DINO (independent of use_2_5d)
        dino_slices = self._load_2_5d_slices(sample)  # [3, H, W] always
        
        # Load Label
        label = self._load_label(sample['label_path'])
        
        # Resize image and label to target_size if specified
        if self.target_size is not None:
            h_target, w_target = self.target_size
            
            # 2.5D: shape [3, H, W]
            if self.use_2_5d:
                _, h_curr, w_curr = image_slice.shape
                if h_curr != h_target or w_curr != w_target:
                    # Resize each channel
                    resized_channels = []
                    for c in range(image_slice.shape[0]):
                        resized_ch = cv2.resize(
                            image_slice[c], (w_target, h_target),
                            interpolation=cv2.INTER_LINEAR
                        )
                        resized_channels.append(resized_ch)
                    image_slice = np.stack(resized_channels, axis=0)  # [3, h_target, w_target]
            else:
                # 2D: shape [H, W]
                if image_slice.shape[0] != h_target or image_slice.shape[1] != w_target:
                    # Image: INTER_LINEAR
                    image_slice = cv2.resize(
                        image_slice, (w_target, h_target),
                        interpolation=cv2.INTER_LINEAR
                    )
            
            # Resize dino_slices [3, H, W]
            _, dh, dw = dino_slices.shape
            if dh != h_target or dw != w_target:
                resized_dino = []
                for c in range(dino_slices.shape[0]):
                    resized_dino.append(
                        cv2.resize(dino_slices[c], (w_target, h_target),
                                   interpolation=cv2.INTER_LINEAR)
                    )
                dino_slices = np.stack(resized_dino, axis=0)
            
            # Resize label
            if label.shape[0] != h_target or label.shape[1] != w_target:
                # Label: INTER_NEAREST (keep binary mask)
                label = cv2.resize(
                    label.astype(np.float32), (w_target, h_target),
                    interpolation=cv2.INTER_NEAREST
                ).astype(np.float32)
        
        data = {
            'image': image_slice,
            'label': label,
            'dino_slices': dino_slices,
            'slice_idx': sample['slice_idx'],
            'side': sample['side'],
            'patient_id': sample['patient_id'],
            'scan_type': sample['scan_type'],
            'mol': sample.get('mol', 0),
            'metadata': sample
        }
        
        # Apply transform
        if self.transform is not None:
            data = self.transform(data)
        
        return data
    
    def _load_2_5d_slices(self, sample: Dict) -> np.ndarray:
        """
        Load adjacent slices to form a 3-channel 2.5D input.
        """
        nifti_basename = sample['nifti_basename']
        curr_idx = sample['slice_idx']
        
        # Current slice
        curr_slice = self._load_npy_slice(sample['slice_npy_path'])
        H, W = curr_slice.shape
        
        # Previous slice (duplicate current if first slice)
        prev_idx = curr_idx - 1
        prev_npy_path = self.data_dir / nifti_basename / f"{prev_idx:03d}.npy"
        if prev_idx >= 0 and prev_npy_path.exists():
            prev_slice = self._load_npy_slice(str(prev_npy_path))
        else:
            prev_slice = curr_slice.copy()
        
        # Next slice (duplicate current if last slice)
        next_idx = curr_idx + 1
        next_npy_path = self.data_dir / nifti_basename / f"{next_idx:03d}.npy"
        if next_npy_path.exists():
            next_slice = self._load_npy_slice(str(next_npy_path))
        else:
            next_slice = curr_slice.copy()
        
        # Resize if shapes mismatch
        if prev_slice.shape != curr_slice.shape:
            prev_slice = cv2.resize(prev_slice, (W, H), interpolation=cv2.INTER_LINEAR)
        if next_slice.shape != curr_slice.shape:
            next_slice = cv2.resize(next_slice, (W, H), interpolation=cv2.INTER_LINEAR)
        
        # Stack: [3, H, W]
        slices_3d = np.stack([prev_slice, curr_slice, next_slice], axis=0)
        return slices_3d
    
    def _load_npy_slice(self, slice_npy_path: str) -> np.ndarray:
        """
        Load and normalize 2D slice from .npy file.
        """
        # Load .npy
        if not os.path.exists(slice_npy_path):
            raise FileNotFoundError(f"Slice .npy not found: {slice_npy_path}")
        
        slice_2d = np.load(slice_npy_path).astype(np.float32)
        
        # Normalization
        if self.normalize == 'minmax':
            # Min-Max normalization to [0, 1]
            vmin = slice_2d.min()
            vmax = slice_2d.max()
            if vmax > vmin:
                slice_2d = (slice_2d - vmin) / (vmax - vmin)
            else:
                slice_2d = np.zeros_like(slice_2d)
        
        elif self.normalize == 'zscore':
            # Z-score normalization
            mean = slice_2d.mean()
            std = slice_2d.std()
            if std > 0:
                slice_2d = (slice_2d - mean) / std
            else:
                slice_2d = slice_2d - mean
        
        # normalize == None: No normalization
        
        return slice_2d
    
    def _load_label(self, label_path: str) -> np.ndarray:
        """
        Load PNG label as binary mask.
        """
        img = Image.open(label_path).convert('L')
        label = np.array(img, dtype=np.float32)
        
        # Convert to binary mask
        label = (label > 0).astype(np.float32)
        
        return label
    
    def get_patient_metadata(self, patient_id: str) -> pd.Series:
        """Get metadata for a specific patient."""
        patient_data = self.metadata[
            self.metadata['Patient ID'].astype(str) == str(patient_id)
        ]
        if len(patient_data) > 0:
            return patient_data.iloc[0]
        return None
    
    def get_nifti_info(self, nifti_path: str) -> Dict:
        """Return metadata info of NIfTI file."""
        nii = nib.load(nifti_path)
        header = nii.header
        zooms = header.get_zooms()[:3]
        
        return {
            'shape': nii.shape,
            'voxel_size': zooms,
            'slice_spacing': zooms[2],
            'affine': nii.affine
        }


def _resize_tensor_2d(x: torch.Tensor, target_h: int, target_w: int, mode: str = 'bilinear') -> torch.Tensor:
    """Resize [C, H, W] tensor to target (H, W)."""
    if x.dim() == 2:
        x = x.unsqueeze(0)
    # x: [C, H, W] -> [1, C, H, W]
    x = x.unsqueeze(0)
    x = torch.nn.functional.interpolate(
        x, size=(target_h, target_w),
        mode='bilinear' if mode == 'bilinear' else 'nearest',
        align_corners=False
    )
    return x.squeeze(0)


def collate_fn_2d(batch: List[Dict]) -> Dict[str, any]:
    """
    Custom collate function for 2D DataLoader.
    Resizes batch items to maximum H and W in the batch if shapes mismatch.
    """
    images = []
    labels = []
    dino_slices_list = []
    slice_indices = []
    sides = []
    patient_ids = []
    scan_types = []
    mol_labels = []
    metadata = []
    
    for item in batch:
        img_np = item['image'].astype(np.float32)
        
        if img_np.ndim == 2:
            # 2D: [H, W] → [1, H, W]
            img = torch.from_numpy(img_np).unsqueeze(0)
        elif img_np.ndim == 3:
            # 2.5D: [3, H, W] → [3, H, W]
            img = torch.from_numpy(img_np)
        else:
            raise ValueError(f"Unexpected image shape: {img_np.shape}")
        
        lbl = torch.from_numpy(item['label'].astype(np.float32)).unsqueeze(0)   # [1, H, W]
        images.append(img)
        labels.append(lbl)
        
        # dino_slices: always [3, H, W]
        if 'dino_slices' in item:
            ds = torch.from_numpy(item['dino_slices'].astype(np.float32))  # [3, H, W]
            dino_slices_list.append(ds)
        
        slice_indices.append(item['slice_idx'])
        sides.append(item['side'])
        patient_ids.append(item['patient_id'])
        scan_types.append(item['scan_type'])
        mol_labels.append(item.get('mol', 0))
        metadata.append(item['metadata'])
    
    # Handle shape mismatch by resizing all to max resolution
    shapes = [(img.shape[1], img.shape[2]) for img in images]
    if len(set(shapes)) > 1:
        target_h = max(s[0] for s in shapes)
        target_w = max(s[1] for s in shapes)
        images = [_resize_tensor_2d(img, target_h, target_w, 'bilinear') for img in images]
        labels = [_resize_tensor_2d(lbl, target_h, target_w, 'nearest') for lbl in labels]
        if dino_slices_list:
            dino_slices_list = [_resize_tensor_2d(ds, target_h, target_w, 'bilinear') for ds in dino_slices_list]
    
    # Stack
    images = torch.stack(images, dim=0)  # [B, 1, H, W] or [B, 3, H, W]
    labels = torch.stack(labels, dim=0)  # [B, 1, H, W]
    
    result = {
        'image': images,
        'label': labels,
        'slice_idx': slice_indices,
        'side': sides,
        'patient_id': patient_ids,
        'scan_type': scan_types,
        'mol': torch.tensor(mol_labels, dtype=torch.long),
        'metadata': metadata
    }
    
    if dino_slices_list:
        result['dino_slices'] = torch.stack(dino_slices_list, dim=0)  # [B, 3, H, W]
    
    return result


if __name__ == "__main__":
    # Test code
    print("=" * 80)
    print("TMJ 2D Dataset Test")
    print("=" * 80)
    
    # Train set
    train_dataset = TMJ_dataset_2D(split='train', normalize='minmax')
    print(f"\nTrain dataset: {len(train_dataset)} 2D slices")
    
    # Val set
    val_dataset = TMJ_dataset_2D(split='val', normalize='minmax')
    print(f"Val dataset: {len(val_dataset)} 2D slices")
    
    # Check sample data
    if len(train_dataset) > 0:
        print("\n" + "=" * 80)
        print("Sample data:")
        print("=" * 80)
        sample = train_dataset[0]
        print(f"Patient ID: {sample['patient_id']}")
        print(f"Scan type: {sample['scan_type']}")
        print(f"Side: {sample['side']}")
        print(f"Slice index: {sample['slice_idx']}")
        print(f"Image shape: {sample['image'].shape}")
        print(f"Image range: [{sample['image'].min():.4f}, {sample['image'].max():.4f}]")
        print(f"Label shape: {sample['label'].shape}")
        print(f"Label sum: {sample['label'].sum():.0f} pixels")
        
        # Statistics
        print("\n" + "=" * 80)
        print("Dataset Statistics:")
        print("=" * 80)
        right_count = sum(1 for s in train_dataset.samples if s['side'] == 'right')
        left_count = sum(1 for s in train_dataset.samples if s['side'] == 'left')
        print(f"Right slices: {right_count}")
        print(f"Left slices: {left_count}")
    
    print("\n" + "=" * 80)
    print("Dataset test completed!")
    print("=" * 80)
