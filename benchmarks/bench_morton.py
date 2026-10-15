"""Reproducible, alternating-order Morton CPU/CUDA microbenchmarks.

Uses all four complete example clouds by default and a frozen reference.
FordExample is prequantized signed millimeters; SemanticKITTIExample is meters.
Run this without other benchmarks competing for CPU/GPU resources.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import io
from src import morton as mt


def load_reference(device):
    spec = importlib.util.spec_from_file_location("reference_morton", ROOT / "benchmarks/reference/src/morton.py")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    # Fix only the table placement, not the frozen reference computation.
    reference._get_occ_bits(device)
    reference._get_expand_coords_base(device)
    return reference


def reference_hierarchy(reference, coords):
    coords, code = reference.morton3_sort(coords)
    result = []
    while True:
        coords, code, occupancy = reference.down_once(coords, code)
        result.append((coords.clone(), occupancy.clone()))
        bitdepth = math.ceil(math.log2(int(coords[:, 1:].max().item()) + 1))
        if bitdepth <= 2:
            break
    return result[::-1]


def assert_equal(left, right):
    if isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_equal(a, b)
    else:
        assert left.dtype == right.dtype and torch.equal(left, right)


def time_pair(baseline, optimized, device, warmup, repeats):
    synchronize = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else lambda: None
    for _ in range(warmup):
        baseline()
        optimized()
    times = [[], []]
    for iteration in range(repeats):
        # Alternate the pair's ordering to reduce cache/thermal ordering bias.
        for index in ((0, 1) if iteration % 2 == 0 else (1, 0)):
            synchronize()
            started = time.perf_counter_ns()
            output = (baseline if index == 0 else optimized)()
            synchronize()
            elapsed = (time.perf_counter_ns() - started) / 1e6
            times[index].append(elapsed)
            del output
    before, after = (statistics.median(values) for values in times)
    return dict(baseline_ms=before, optimized_ms=after, speedup=before / after,
                baseline_samples_ms=times[0], optimized_samples_ms=times[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--posq", type=int, default=16)
    parser.add_argument("--prequantized", choices=("auto", "yes", "no"), default="auto",
                        help="auto treats the repository's FordExample as prequantized, KITTI as meters")
    parser.add_argument("--files", nargs="*", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/results_morton.json")
    args = parser.parse_args()
    if args.threads < 1 or args.repeats < 1 or args.warmup < 0 or args.channels < 1 or args.posq < 1:
        parser.error("threads, repeats, channels and posq must be positive; warmup must be nonnegative")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    reference = load_reference(device)
    files = args.files if args.files is not None else sorted((ROOT / "data").glob("*Example/*.ply"))
    report = dict(torch_version=torch.__version__, python_version=platform.python_version(),
                  platform=platform.platform(), device=str(device), threads=args.threads,
                  warmup=args.warmup, repeats=args.repeats, channels=args.channels,
                  posq=args.posq, prequantized=args.prequantized,
                  timer="perf_counter_ns; median; alternating pair order; CUDA synchronize when applicable",
                  samples=[])
    with torch.inference_mode():
        for path in files:
            points = io.read_points(path)
            prequantized = ("FordExample" in path.parts if args.prequantized == "auto"
                            else args.prequantized == "yes")
            values = torch.tensor(points if prequantized else points / 0.001 + 131072)
            if prequantized and values.min() < 0:
                values += 131072
            xyz = torch.round(values / args.posq).int()
            if xyz.min() < 0 or xyz.max() >= (1 << 21):
                raise ValueError(f"quantized coordinates outside Morton range: {path}")
            coords = torch.cat((xyz[:, :1] * 0, xyz), dim=1).to(device)
            sorted_c, sorted_code = mt.morton3_sort(coords)
            parent_c, _, occupancy = mt.down_once(sorted_c, sorted_code)
            generator = torch.Generator(device=device).manual_seed(11)
            features = torch.randn((len(parent_c), args.channels), device=device, generator=generator)
            stages = {
                "morton3_code": (lambda: reference.morton3_code(coords[:, 1:]),
                                 lambda: mt.morton3_code(coords[:, 1:])),
                "morton3_sort": (lambda: reference.morton3_sort(coords),
                                 lambda: mt.morton3_sort(coords)),
                "down_once": (lambda: reference.down_once(sorted_c, sorted_code),
                              lambda: mt.down_once(sorted_c, sorted_code)),
                "build_hierarchy": (lambda: reference_hierarchy(reference, coords),
                                    lambda: mt.build_hierarchy(coords)),
                "upscale_feature": (lambda: reference.upscale_feature(occupancy, features),
                                    lambda: mt.upscale_feature(occupancy, features)),
                "upscale_coordinate": (lambda: reference.upscale_coordinate(parent_c, occupancy),
                                       lambda: mt.upscale_coordinate(parent_c, occupancy)),
                "upscale_coordinate_feature": (
                    lambda: reference.upscale_coordinate_feature(parent_c, occupancy, features),
                    lambda: mt.upscale_coordinate_feature(parent_c, occupancy, features)),
            }
            sample = dict(file=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                          source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          prequantized=prequantized,
                          quantized_coords_sha256=hashlib.sha256(coords.cpu().numpy().tobytes()).hexdigest(),
                          input_points=len(coords), unique_voxels=len(sorted_c), parent_voxels=len(parent_c),
                          hierarchy_levels=len(mt.build_hierarchy(coords)), stages={})
            for name, (baseline, optimized) in stages.items():
                assert_equal(baseline(), optimized())
                result = time_pair(baseline, optimized, device, args.warmup, args.repeats)
                result["forward_exact"] = True
                sample["stages"][name] = result
                print(f"{sample['file']} {name}: {result['baseline_ms']:.3f} -> "
                      f"{result['optimized_ms']:.3f} ms ({result['speedup']:.2f}x)", flush=True)
            report["samples"].append(sample)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
