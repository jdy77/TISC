"""
NIfTI → per-slice .npy converter

Original NIfTI:
    /path/to/data/nifti

Conversion result:
    /path/to/data_converted/nifti/<nifti_basename>/<zzz>.npy

Where <zzz> is a 3-digit zero-padded z-slice index (e.g., 000.npy).
Saves slices using z-index directly, matching `TMJ_dataset_2D` label naming convention.
e.g., label `<basename>_005.png` matches slice `<basename>/005.npy`.
"""

import os
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm


SRC_DIR = Path(
    "/path/to/data/nifti"
)
DST_ROOT = Path(
    "/path/to/data_converted/nifti"
)


def get_nifti_basename(nifti_path: Path) -> str:
    """
    Return base name of NIfTI file by removing extensions (.nii or .nii.gz).
    """
    name = nifti_path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return nifti_path.stem


def convert_one_nifti(nifti_path: Path, dst_root: Path) -> None:
    """
    Read a single NIfTI file and save z-axis slices as .npy files.
    """
    basename = get_nifti_basename(nifti_path)
    dst_dir = dst_root / basename
    dst_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nConverting: {nifti_path}")
    print(f"  -> output dir: {dst_dir}")

    # Load NIfTI
    nii = nib.load(str(nifti_path))
    volume = nii.get_fdata().astype(np.float32)  # [H, W, Z] (assumed)

    if volume.ndim != 3:
        raise ValueError(
            f"Expected 3D volume for {nifti_path}, but got shape {volume.shape}"
        )

    h, w, z_dim = volume.shape
    print(f"  volume shape: (H={h}, W={w}, Z={z_dim})")

    # Save z slices as 3-digit zero-padded .npy (e.g. 000.npy)
    for z in tqdm(range(z_dim), desc=f"Slices ({basename})"):
        slice_2d = volume[:, :, z]  # [H, W], float32
        out_name = f"{z:03d}.npy"
        out_path = dst_dir / out_name

        # Overwrite if exists.
        np.save(out_path, slice_2d)


def main():
    if not SRC_DIR.exists():
        raise FileNotFoundError(f"Source directory not found: {SRC_DIR}")

    DST_ROOT.mkdir(parents=True, exist_ok=True)

    # Process both .nii and .nii.gz
    nifti_files = sorted(
        list(SRC_DIR.glob("*.nii")) + list(SRC_DIR.glob("*.nii.gz"))
    )

    if not nifti_files:
        print(f"No NIfTI files found in {SRC_DIR}")
        return

    print(f"Found {len(nifti_files)} NIfTI files in {SRC_DIR}")
    print(f"Output root: {DST_ROOT}")

    for nifti_path in nifti_files:
        convert_one_nifti(nifti_path, DST_ROOT)

    print("\nAll conversions completed.")


if __name__ == "__main__":
    main()

