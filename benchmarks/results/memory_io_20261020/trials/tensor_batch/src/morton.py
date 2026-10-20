"""Morton ordered octree utilities for nonnegative, at most 21-bit voxels.

The fast sorting path accepts one batch, as used by ELiC's batch-size-one
training and codec. Pass ``batch_mode="multiple"`` explicitly for multiple
batches; it keeps separate octrees in batch-major, then Morton order. Batch
identifiers are preserved by downsampling and upsampling.
"""
from typing import List, Tuple

import torch


@torch.no_grad()
def _dilate3_u64(v: torch.Tensor) -> torch.Tensor:
    """Spread each of the low 21 bits into every third bit of int64."""
    v = v.to(torch.int64) & 0x1fffff
    # The initial mask owns storage even for int64 input. Subsequent in-place
    # operations therefore cannot mutate xyz and avoid two temporaries/stage.
    v.bitwise_or_(v << 32).bitwise_and_(0x1f00000000ffff)
    v.bitwise_or_(v << 16).bitwise_and_(0x1f0000ff0000ff)
    v.bitwise_or_(v << 8).bitwise_and_(0x100f00f00f00f00f)
    v.bitwise_or_(v << 4).bitwise_and_(0x10c30c30c30c30c3)
    return v.bitwise_or_(v << 2).bitwise_and_(0x1249249249249249)


@torch.no_grad()
def morton3_code(xyz: torch.Tensor) -> torch.Tensor:
    """Interleave x, y, z into an int64 Morton code; xyz must be in [0, 2**21)."""
    if xyz.device.type == "cpu":
        # Batch adjacent XYZ loads, then work on contiguous axis rows. Bound
        # the working set instead of streaming three full-cloud temporaries
        # through memory at every bit-spreading step.
        block = 32768
        if xyz.shape[0] <= block:
            return _morton3_code_cpu_block(xyz)
        code = torch.empty(xyz.shape[0], dtype=torch.int64, device=xyz.device)
        for start in range(0, xyz.shape[0], block):
            code[start:start + block].copy_(_morton3_code_cpu_block(xyz[start:start + block]))
        return code
    x, y, z = xyz.unbind(dim=1)
    code = _dilate3_u64(x)
    code.bitwise_or_(_dilate3_u64(y).bitwise_left_shift_(1))
    return code.bitwise_or_(_dilate3_u64(z).bitwise_left_shift_(2))


def _morton3_code_cpu_block(xyz: torch.Tensor) -> torch.Tensor:
    # Own storage even for int64 input; no operation may mutate coordinates.
    spread = xyz.T.to(dtype=torch.int64, copy=True, memory_format=torch.contiguous_format)
    spread.bitwise_and_(0x1fffff)
    for shift, mask in ((32, 0x1f00000000ffff), (16, 0x1f0000ff0000ff),
                        (8, 0x100f00f00f00f00f), (4, 0x10c30c30c30c30c3),
                        (2, 0x1249249249249249)):
        spread.bitwise_or_(spread << shift).bitwise_and_(mask)
    x, y, z = spread.unbind(dim=0)
    # Return independent contiguous storage, without retaining the 3-axis buffer.
    code = x | y.bitwise_left_shift_(1)
    return code.bitwise_or_(z.bitwise_left_shift_(2))


def _check_coords(bxyz: torch.Tensor) -> None:
    if bxyz.ndim != 2 or bxyz.shape[1] != 4:
        raise ValueError("coordinates must have shape (N, 4): [batch, x, y, z]")
    if bxyz.dtype not in (torch.int32, torch.int64):
        raise TypeError("coordinates must be int32 or int64")


@torch.no_grad()
def morton3_sort(bxyz: torch.Tensor, *, batch_mode: str = "single"
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deduplicate and sort integer [batch, x, y, z] coordinates.

    ``batch_mode="single"`` is the synchronization-free fast path. Its input
    must contain a single batch ID and coordinates in [0, 2**21); these value
    preconditions are intentionally not checked with a device-to-host read.
    ``batch_mode="multiple"`` supports int32-range batch IDs with a second stable
    sort. Both modes preserve IDs and remove duplicate rows. The single-batch
    output is elementwise identical to the original implementation.
    """
    _check_coords(bxyz)
    if batch_mode not in ("single", "multiple"):
        raise ValueError("batch_mode must be 'single' or 'multiple'")
    code = morton3_code(bxyz[:, 1:])
    code, order = torch.sort(code, stable=(batch_mode == "multiple"))
    bxyz = bxyz.index_select(0, order)
    if batch_mode == "multiple":
        order = torch.argsort(bxyz[:, 0], stable=True)
        bxyz = bxyz.index_select(0, order)
        code = code.index_select(0, order)
    keep = torch.ones(code.shape, dtype=torch.bool, device=code.device)
    keep[1:] = (code[1:] != code[:-1]) | (bxyz[1:, 0] != bxyz[:-1, 0])
    return bxyz[keep], code[keep]


@torch.no_grad()
def down_once(bxyz: torch.Tensor, code: torch.Tensor
              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group unique sorted children into parents and their 8-bit occupancy.

    Input must come from :func:`morton3_sort` or an earlier downsampling level.
    Batch-major Morton order supports multiple batches without mixing them.
    Parent coordinates/occupancy are int32, matching the original interface.
    """
    parent_code_all = code >> 3
    first = torch.ones(code.shape, dtype=torch.bool, device=code.device)
    first[1:] = ((parent_code_all[1:] != parent_code_all[:-1]) |
                 (bxyz[1:, 0] != bxyz[:-1, 0]))
    first_idx = first.nonzero(as_tuple=True)[0]
    inv = first.to(torch.int64).cumsum(0) - 1
    parent_code = parent_code_all.index_select(0, first_idx)
    # Morton code's three low bits already contain the child octant.
    occ_bits = (1 << (code & 7)).to(torch.int32)
    occ_sym = torch.zeros(first_idx.shape, dtype=torch.int32, device=bxyz.device)
    occ_sym.scatter_add_(0, inv, occ_bits)
    # Gather before shifting: only parent-sized coordinate work is necessary.
    parent_bxyz = bxyz.index_select(0, first_idx).to(torch.int32)
    parent_bxyz[:, 1:] >>= 1
    return parent_bxyz, parent_code, occ_sym


@torch.no_grad()
def build_hierarchy(bxyz: torch.Tensor, stop_bitdepth: int = 2, *,
                    batch_mode: str = "single"
                    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Return coarse-to-fine (parent coordinates, occupancy) octree levels.

    Read the maximum spatial coordinate once to calculate the number of levels;
    all subsequent grouping stays on the input device. As in ELiC's original
    loop, nonempty shallow clouds still undergo at least one downsampling.
    Empty input returns an empty list. Returned tensors own their level's
    storage, so intermediate ``clone`` calls are unnecessary.
    """
    _check_coords(bxyz)
    if not isinstance(stop_bitdepth, int) or stop_bitdepth < 0:
        raise ValueError("stop_bitdepth must be a nonnegative integer")
    if bxyz.shape[0] == 0:
        return []
    max_coord = int(bxyz[:, 1:].max().item())
    if max_coord < 0 or max_coord >= (1 << 21):
        raise ValueError("spatial coordinates must be in [0, 2**21)")
    levels = max(1, max_coord.bit_length() - stop_bitdepth)
    bxyz, code = morton3_sort(bxyz, batch_mode=batch_mode)
    hierarchy = []
    for _ in range(levels):
        bxyz, code, occupancy = down_once(bxyz, code)
        hierarchy.append((bxyz, occupancy))
    hierarchy.reverse()
    return hierarchy


def _device_key(device=None) -> torch.device:
    # Resolve an unspecified CUDA ordinal before using it as a cache key.
    d = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
    if d.type == "cuda" and d.index is None:
        d = torch.device("cuda", torch.cuda.current_device())
    return d


@torch.no_grad()
def _get_occ_bits(device=None) -> torch.Tensor:
    """Return a per-device cached (256, 8) occupancy lookup table."""
    d = _device_key(device)
    if not hasattr(_get_occ_bits, "_cache"):
        _get_occ_bits._cache = {}
    if d not in _get_occ_bits._cache:
        occ = torch.arange(256, dtype=torch.int32, device=d)[:, None]
        pos = torch.arange(8, dtype=torch.int32, device=d)[None, :]
        _get_occ_bits._cache[d] = ((occ >> pos) & 1).bool()
    return _get_occ_bits._cache[d]


@torch.no_grad()
def _get_expand_coords_base(device=None) -> torch.Tensor:
    """Return per-device cached (8, 3) child offsets in Morton order."""
    d = _device_key(device)
    if not hasattr(_get_expand_coords_base, "_cache"):
        _get_expand_coords_base._cache = {}
    if d not in _get_expand_coords_base._cache:
        pos = torch.arange(8, dtype=torch.int32, device=d)
        _get_expand_coords_base._cache[d] = torch.stack(
            (pos & 1, (pos >> 1) & 1, (pos >> 2) & 1), dim=1)
    return _get_expand_coords_base._cache[d]


def _occupied_children(x_O: torch.Tensor, device: torch.device
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    bits = _get_occ_bits(device)[x_O.reshape(-1).to(device=device, dtype=torch.long)]
    return bits.nonzero(as_tuple=True)


def upscale_feature(x_O: torch.Tensor, x_F: torch.Tensor) -> torch.Tensor:
    """Gather features only for occupied children, preserving autograd."""
    parents, _ = _occupied_children(x_O, x_F.device)
    return x_F.index_select(0, parents)


@torch.no_grad()
def _upscale_coords(x_C: torch.Tensor, parents: torch.Tensor,
                    children: torch.Tensor) -> torch.Tensor:
    coords = x_C.index_select(0, parents).to(torch.int32)
    coords[:, 1:] *= 2
    coords[:, 1:] += _get_expand_coords_base(x_C.device).index_select(0, children)
    return coords


@torch.no_grad()
def upscale_coordinate(x_C: torch.Tensor, x_O: torch.Tensor) -> torch.Tensor:
    """Expand occupied child coordinates and preserve each parent's batch ID."""
    parents, children = _occupied_children(x_O, x_C.device)
    return _upscale_coords(x_C, parents, children)


def upscale_coordinate_feature(x_C: torch.Tensor, x_O: torch.Tensor,
                               x_F: torch.Tensor
                               ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Expand occupied coordinates/features once; feature gradients are kept."""
    if x_C.device != x_F.device:
        raise ValueError("coordinates and features must be on the same device")
    parents, children = _occupied_children(x_O, x_C.device)
    return _upscale_coords(x_C, parents, children), x_F.index_select(0, parents)
