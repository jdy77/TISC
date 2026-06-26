import torch
import torch.nn.functional as F

from .model_unet_all_mol_learnable_query import (
    UNetMOLLearnableQuery,
    build_unet_all_mol_learnable_query,
    mol_pointrend_loss_lq,
    mol_warmup_loss_lq,
    point_sample,
    sampling_points,
)


def sample_point_labels(labels: torch.Tensor, point_coords: torch.Tensor) -> torch.Tensor:
    """Sample GT labels on point coordinates [0,1] using nearest interpolation."""
    if point_coords.dim() != 3 or point_coords.shape[-1] != 2:
        raise ValueError(f"point_coords must be [B, N, 2], got {tuple(point_coords.shape)}")
    coords = point_coords * 2.0 - 1.0
    sampled = F.grid_sample(labels.float(), coords.unsqueeze(2), mode="nearest", align_corners=False)
    return sampled.squeeze(3)


def point_sample_pr(input: torch.Tensor, point_coords: torch.Tensor, **kwargs) -> torch.Tensor:
    """Backward-compatible alias used by train script."""
    return point_sample(input, point_coords, **kwargs)


__all__ = [
    "UNetMOLLearnableQuery",
    "build_unet_all_mol_learnable_query",
    "mol_pointrend_loss_lq",
    "mol_warmup_loss_lq",
    "point_sample",
    "point_sample_pr",
    "sampling_points",
    "sample_point_labels",
    "get_model",
]


def get_model(model_name, **kwargs):
    """Get model by name. Only learnable-query model is supported."""
    if model_name != "unet_all_mol_learnable_query":
        raise ValueError(
            f"Unknown model: {model_name}. "
            "Only 'unet_all_mol_learnable_query' is available after cleanup."
        )

    return build_unet_all_mol_learnable_query(
        n_channels=kwargs.get("n_channels", 1),
        n_classes=kwargs.get("n_classes", 1),
        bilinear=kwargs.get("bilinear", False),
        train_num_points=kwargs.get("train_num_points", 1024),
        inference_num_points=kwargs.get("inference_num_points", 2048),
        dinov3_model=kwargs.get("dinov3_model", None),
        dinov3_feature_dim=kwargs.get("dinov3_feature_dim", 768),
        dinov3_input_size=kwargs.get("dinov3_input_size", 518),
        main_slice_weight=kwargs.get("main_slice_weight", 1.0),
        learnable_query_ckpt=kwargs.get("learnable_query_ckpt", ""),
    )
