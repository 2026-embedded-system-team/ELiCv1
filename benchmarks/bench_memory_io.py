"""Compare the frozen pre-memory-I/O working tree with selected source files.

The CPU default uses one PyTorch thread, seed 11, three warmups and eleven
alternating-order pairs. Timings include returned-output allocation on both
sides; input generation, PLY parsing, quantization and correctness checks are
outside the measured region. No files under reports/ are written.

The archived tensor batching trial was rejected because aggregate real-cloud
hierarchy time showed no stable gain. To reproduce it from the repository root:
python benchmarks/bench_memory_io.py --after-dir benchmarks/results/memory_io_20261020/trials/tensor_batch/src --output benchmarks/results/memory_io_20261020/trials/tensor_batch/reproduced_tensor_benchmark.json
Omitting --after-dir compares against the current production src/ directory.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks/results/memory_io_20261020"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def path_label(path):
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def assert_equal(left, right):
    if isinstance(left, (list, tuple)):
        assert isinstance(right, type(left)) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_equal(a, b)
    else:
        assert left.dtype == right.dtype and left.shape == right.shape
        assert torch.equal(left, right), "before/after output differs"


def percentile(values, fraction):
    ordered = sorted(values)
    location = (len(ordered) - 1) * fraction
    low = int(location)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (location - low)


def timing_stats(values):
    return {
        "median_ms": statistics.median(values),
        "p10_ms": percentile(values, 0.1),
        "p90_ms": percentile(values, 0.9),
        "samples_ms": values,
    }


def measure_pair(before, after, device, warmup, repeats):
    synchronize = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else lambda: None
    functions = (before, after)
    for iteration in range(warmup):
        for index in ((0, 1) if iteration % 2 == 0 else (1, 0)):
            result = functions[index]()
            synchronize()
            del result
    samples = [[], []]
    orders = []
    for iteration in range(repeats):
        order = (0, 1) if iteration % 2 == 0 else (1, 0)
        orders.append(["before" if index == 0 else "after" for index in order])
        for index in order:
            synchronize()
            started = time.perf_counter_ns()
            result = functions[index]()
            synchronize()
            samples[index].append((time.perf_counter_ns() - started) / 1e6)
            del result
    baseline, optimized = map(timing_stats, samples)
    return {
        "before": baseline,
        "after": optimized,
        "speedup": baseline["median_ms"] / optimized["median_ms"],
        "paired_speedups": [a / b for a, b in zip(*samples)],
        "pair_order": orders,
    }


def run_case(report, args, device, label, group, stage, before, after, inputs, **metadata):
    snapshots = [tensor.clone() for tensor in inputs]
    expected = before()
    for tensor, snapshot in zip(inputs, snapshots):
        assert torch.equal(tensor, snapshot), f"before mutates input: {label}"
    actual = after()
    assert_equal(expected, actual)
    del expected, actual
    for tensor, snapshot in zip(inputs, snapshots):
        assert torch.equal(tensor, snapshot), f"after mutates input: {label}"
    case = {
        "label": label,
        "group": group,
        "stage": stage,
        **metadata,
        "exact_equal": True,
        "inputs_unchanged": True,
    }
    if not args.correctness_only:
        case.update(measure_pair(before, after, device, args.warmup, args.repeats))
    for tensor, snapshot in zip(inputs, snapshots):
        assert torch.equal(tensor, snapshot), f"timed call mutates input: {label}"
    report["cases"].append(case)
    if args.correctness_only:
        print(f"{label:<67} exact, inputs unchanged", flush=True)
    else:
        print(f"{label:<67} {case['before']['median_ms']:10.4f} "
              f"{case['after']['median_ms']:10.4f} {case['speedup']:8.3f}x", flush=True)


def save_results(report, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    columns = ["label", "group", "stage", "rows", "length", "needs_normalization", "file",
               "before_median_ms", "before_p10_ms", "before_p90_ms", "after_median_ms",
               "after_p10_ms", "after_p90_ms", "speedup", "exact_equal", "inputs_unchanged"]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for case in report["cases"]:
            row = {name: case[name] for name in columns if name in case}
            for variant in ("before", "after"):
                if variant in case:
                    row.update({f"{variant}_{stat}": case[variant][stat]
                                for stat in ("median_ms", "p10_ms", "p90_ms")})
            writer.writerow(row)
    print(f"Saved {output}\nSaved {csv_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--sizes", nargs="+", type=int, default=[64, 4096, 100000, 400000])
    parser.add_argument("--posq", type=int, default=16)
    parser.add_argument("--files", nargs="+", type=Path)
    parser.add_argument("--skip-real", action="store_true")
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--skip-cdf", action="store_true")
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--before-dir", type=Path, default=RESULTS / "before/src")
    parser.add_argument("--after-dir", type=Path, default=ROOT / "src",
                        help="source directory containing the candidate morton.py and op.py")
    parser.add_argument("--output", type=Path, default=RESULTS / "tensor_benchmark.json")
    args = parser.parse_args()
    if args.threads < 1 or args.warmup < 0 or args.repeats < 1 or args.posq < 1 or min(args.sizes) < 1:
        parser.error("threads, repeats, posq and sizes must be positive; warmup must be nonnegative")
    # Reject report paths even when overridden: this experiment stays separate.
    output_path = args.output.resolve()
    if output_path.is_relative_to(ROOT) and "reports" in output_path.relative_to(ROOT).parts:
        parser.error("write experiment results outside reports/")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    source_paths = {
        "before_morton": args.before_dir / "morton.py",
        "before_op": args.before_dir / "op.py",
        "after_morton": args.after_dir / "morton.py",
        "after_op": args.after_dir / "op.py",
        "point_cloud_io": ROOT / "src/io.py",
        "benchmark": Path(__file__).resolve(),
    }
    source_hashes = {name: {"path": path_label(path), "sha256": sha256(path)}
                     for name, path in source_paths.items()}
    before_mt = load_module("memory_io_before_morton", source_paths["before_morton"])
    after_mt = load_module("memory_io_after_morton", source_paths["after_morton"])
    before_op = load_module("memory_io_before_op", source_paths["before_op"])
    after_op = load_module("memory_io_after_op", source_paths["after_op"])
    point_io = load_module("memory_io_points", source_paths["point_cloud_io"])
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": "frozen working-tree state before this memory-I/O experiment",
        "environment": {
            "python": platform.python_version(), "python_executable": sys.executable,
            "platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", ""),
            "logical_cpus": os.cpu_count(), "torch": torch.__version__,
            "torch_config": torch.__config__.show(), "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "torch_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        },
        "settings": {"seed": args.seed, "warmup": args.warmup, "repeats": args.repeats,
                     "before_dir": path_label(args.before_dir.resolve()),
                     "after_dir": path_label(args.after_dir.resolve()),
                     "posq": args.posq, "sizes": args.sizes, "correctness_only": args.correctness_only,
                     "timer": "perf_counter_ns; alternating paired order; CUDA synchronization if applicable",
                     "timed_allocation": True,
                     "excluded": ["input creation", "PLY reading", "quantization", "correctness checks"],
                     "synthetic_morton_layout": "xyz view of int32 [batch,x,y,z], row stride 4",
                     "cdf_dtype": "float32", "cdf_length": 17,
                     "cdf_normalization": [True, False]},
        "source_files": source_hashes, "clouds": [], "cases": [],
    }
    print(f"{'case':<67} {'before ms':>10} {'after ms':>10} {'speedup':>9}", flush=True)
    generator = torch.Generator().manual_seed(args.seed)
    with torch.inference_mode():
        if not args.skip_synthetic:
            for rows in args.sizes:
                coords = torch.randint(0, 1 << 21, (rows, 4), dtype=torch.int32, generator=generator).to(device)
                coords[:, 0] = 0
                xyz = coords[:, 1:]
                run_case(report, args, device, f"synthetic/{rows}/morton3_code", "synthetic_morton",
                         "morton3_code", lambda: before_mt.morton3_code(xyz),
                         lambda: after_mt.morton3_code(xyz), [coords], rows=rows)
        if not args.skip_cdf:
            for rows in args.sizes:
                cdf = torch.rand((rows, 17), generator=generator).sort(dim=-1).values.to(device)
                cdf[:, 0] = 0
                cdf[:, -1] = 1
                for normalized in (True, False):
                    run_case(report, args, device, f"cdf/{rows}/normalize={normalized}", "cdf",
                             "convert_to_int_and_normalize",
                             lambda: before_op._convert_to_int_and_normalize(cdf, normalized),
                             lambda: after_op._convert_to_int_and_normalize(cdf, normalized),
                             [cdf], rows=rows, length=17, needs_normalization=normalized)
        if not args.skip_real:
            files = args.files if args.files is not None else sorted((ROOT / "data").glob("*Example/*.ply"))
            if not files:
                raise ValueError("no real input clouds found; supply --files or --skip-real")
            for path in files:
                path = path.resolve()
                points = point_io.read_points(path)
                prequantized = "FordExample" in path.parts
                # Match bench_morton.py: Ford signed millimeters, KITTI meters.
                values = torch.tensor(points if prequantized else points / 0.001 + 131072)
                if prequantized and values.min() < 0:
                    values += 131072
                xyz = torch.round(values / args.posq).int()
                if xyz.min() < 0 or xyz.max() >= 1 << 21:
                    raise ValueError(f"quantized coordinates outside Morton range: {path}")
                coords = torch.cat((xyz[:, :1] * 0, xyz), dim=1).to(device)
                sorted_c, sorted_code = before_mt.morton3_sort(coords)
                cloud_name = path_label(path)
                cloud = {"file": cloud_name, "source_sha256": sha256(path),
                         "quantized_coords_sha256": hashlib.sha256(coords.cpu().numpy().tobytes()).hexdigest(),
                         "input_points": len(coords), "unique_voxels": len(sorted_c),
                         "prequantized": prequantized,
                         "hierarchy_levels": len(before_mt.build_hierarchy(coords))}
                report["clouds"].append(cloud)
                stages = {
                    "morton3_code": (lambda: before_mt.morton3_code(coords[:, 1:]),
                                     lambda: after_mt.morton3_code(coords[:, 1:]), [coords]),
                    "morton3_sort": (lambda: before_mt.morton3_sort(coords),
                                     lambda: after_mt.morton3_sort(coords), [coords]),
                    "down_once": (lambda: before_mt.down_once(sorted_c, sorted_code),
                                  lambda: after_mt.down_once(sorted_c, sorted_code), [sorted_c, sorted_code]),
                    "build_hierarchy": (lambda: before_mt.build_hierarchy(coords),
                                        lambda: after_mt.build_hierarchy(coords), [coords]),
                }
                for stage, (before, after, inputs) in stages.items():
                    run_case(report, args, device, f"{cloud_name}/{stage}", "real_cloud", stage,
                             before, after, inputs, rows=len(coords), file=cloud_name)
                hierarchy = report["cases"][-1]
                if not args.correctness_only:
                    cloud["total_hierarchy_before_ms"] = hierarchy["before"]["median_ms"]
                    cloud["total_hierarchy_after_ms"] = hierarchy["after"]["median_ms"]
                    cloud["total_hierarchy_speedup"] = hierarchy["speedup"]
    for name, path in source_paths.items():
        if sha256(path) != source_hashes[name]["sha256"]:
            raise RuntimeError(f"Source changed during benchmark; rerun after edits finish: {path}")
    save_results(report, args.output.resolve())


if __name__ == "__main__":
    main()
