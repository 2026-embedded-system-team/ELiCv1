"""Pure-PyTorch occupancy statistics for BoE center training.

These functions optimize the offline histogram collection step. They do not
change the trained model, clustering algorithm, or inference-time selection.
"""
from typing import Tuple

import torch

from . import morton


@torch.no_grad()
def occupancy_histogram(occupancy: torch.Tensor) -> torch.Tensor:
    """Return the normalized float32 [low-nibble 16 bins, high-nibble 16 bins].

    Occupancy values must be integers in [0, 255]. Counts and normalization
    match the original BoE center-training script exactly for nonempty input.
    Counts are invariant to voxel ordering, so coordinate sorting is omitted.
    Empty input returns 32 zeros.
    """
    if occupancy.numel() == 0:
        return torch.zeros(32, dtype=torch.float32, device=occupancy.device)
    values = occupancy.reshape(-1).int()
    low = torch.bincount(values & 15, minlength=16)
    high = torch.bincount(values >> 4, minlength=16)
    histogram = torch.cat((low, high)).float()
    return histogram / histogram.sum()


@torch.no_grad()
def collect_histograms(bxyz: torch.Tensor, stop_bitdepth: int = 7, *,
                       batch_mode: str = "single"
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collect octree occupancy histograms and legacy bit-depth labels.

    Returns float32 histograms of shape (levels, 32) and int32 labels of shape
    (levels, 1), both on the coordinate device and in coarse-to-fine order.
    The first label is always ``stop_bitdepth`` even for a shallower cloud,
    preserving the original script's unconditional starting label of seven.
    See ``morton3_sort`` for single/multiple batch input preconditions.
    """
    hierarchy = morton.build_hierarchy(bxyz, stop_bitdepth=stop_bitdepth,
                                        batch_mode=batch_mode)
    if not hierarchy:
        return (torch.empty((0, 32), dtype=torch.float32, device=bxyz.device),
                torch.empty((0, 1), dtype=torch.int32, device=bxyz.device))
    histograms = torch.stack([occupancy_histogram(occ) for _, occ in hierarchy])
    labels = torch.arange(stop_bitdepth, stop_bitdepth + len(hierarchy),
                          dtype=torch.int32, device=bxyz.device).unsqueeze(1)
    return histograms, labels
