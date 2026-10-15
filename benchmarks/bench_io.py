"""Compare frozen and optimized I/O with real samples and reproducible timing.

Run from the repository root: python benchmarks/bench_io.py --repeats 7
The original open3d object construction has no effect on its file output and is
stubbed out: its unavailable optional dependency is excluded from baseline time.
Every timed write targets a new file, avoiding the original shell-rm bug.
"""

import argparse
import ast
import csv
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import io  # noqa: E402


def reference_functions():
    path = ROOT / "benchmarks" / "reference" / "src" / "io.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in ("read_points", "save_ply_ascii_geo")]
    namespace = {
        "os": os, "np": np,
        "o3d": SimpleNamespace(
            geometry=SimpleNamespace(PointCloud=SimpleNamespace),
            utility=SimpleNamespace(Vector3dVector=lambda values: values),
        ),
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["read_points"], namespace["save_ply_ascii_geo"]


def compare(name, sample, old, new, repeats, warmups):
    for _ in range(warmups):
        old()
        new()
    times = {"baseline": [], "optimized": []}
    for repeat in range(repeats):
        order = (("baseline", old), ("optimized", new))
        if repeat % 2:
            order = order[::-1]
        for label, function in order:
            start = time.perf_counter_ns()
            function()
            times[label].append((time.perf_counter_ns() - start) / 1e6)
    baseline_ms = float(np.median(times["baseline"]))
    optimized_ms = float(np.median(times["optimized"]))
    result = {
        "module": name, "sample": sample,
        "baseline_ms": baseline_ms, "optimized_ms": optimized_ms,
        "speedup": baseline_ms / optimized_ms,
        "baseline_p10_ms": float(np.percentile(times["baseline"], 10)),
        "baseline_p90_ms": float(np.percentile(times["baseline"], 90)),
        "optimized_p10_ms": float(np.percentile(times["optimized"], 10)),
        "optimized_p90_ms": float(np.percentile(times["optimized"], 90)),
        "raw_ms": times,
    }
    print(f"{name:22s} {sample:45s} {baseline_ms:8.3f} -> {optimized_ms:8.3f} ms  {result['speedup']:.3f}x", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "io_benchmark.json")
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 0:
        parser.error("repeats must be positive and warmups nonnegative")
    old_read, old_write = reference_functions()
    samples = sorted((ROOT / "data").glob("*Example/*.ply"))
    if not samples:
        raise RuntimeError("No example PLY inputs were found")
    results = []
    sample_points = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # A workspace-local temp directory also works in restricted Windows sessions.
    with tempfile.TemporaryDirectory(prefix="io_bench_", dir=args.output.parent) as temporary:
        folder = Path(temporary)
        sequence = iter(range(10**9))
        for path in samples:
            label = str(path.relative_to(ROOT))
            points = io.read_points(path)
            sample_points[label] = len(points)
            np.testing.assert_array_equal(points, old_read(path))
            results.append(compare("PLY read", label, lambda: old_read(path), lambda: io.read_points(path), args.repeats, args.warmups))

            # Separate names prevent repeat-write behavior from biasing the result.
            def write_with(function):
                function(points, folder / f"cloud_{next(sequence)}.ply")

            old_path = folder / f"baseline_check_{next(sequence)}.ply"
            new_path = folder / f"optimized_check_{next(sequence)}.ply"
            old_write(points, old_path)
            io.save_ply_ascii_geo(points, new_path)
            np.testing.assert_array_equal(io.read_points(old_path).astype(np.float32), io.read_points(new_path).astype(np.float32))
            results.append(compare("PLY write", label, lambda: write_with(old_write), lambda: write_with(io.save_ply_ascii_geo), args.repeats, args.warmups))

        results.append(compare(
            "Batch read sequential", "all four real samples",
            lambda: [old_read(path) for path in samples],
            lambda: io.read_point_clouds(samples, workers=1),
            args.repeats, args.warmups,
        ))
        results.append(compare(
            "Batch read auto", "all four real samples",
            lambda: [old_read(path) for path in samples],
            lambda: io.read_point_clouds(samples), args.repeats, args.warmups,
        ))

    report = {
        "environment": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "cpu_count": os.cpu_count()},
        "repeats": args.repeats, "warmups": args.warmups,
        "sample_points": sample_points,
        "notes": [
            "Wall-clock timings include file open, parsing/formatting and close; files are OS-cached, without fsync.",
            "Baseline functions are compiled unchanged from the frozen AST; unused open3d conversion is stubbed out.",
            "Baseline batch reader uses sequential read_points: original Pool(64) is not a valid Windows script baseline without a main guard.",
            "New output uses nine significant digits to preserve float32 values; ASCII text and file sizes may differ from baseline.",
            "The output is an I/O module benchmark, not complete neural codec or GPU performance.",
        ],
        "results": results,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    rows = [{key: value for key, value in result.items() if key != "raw_ms"} for result in results]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {args.output} and {csv_path}")


if __name__ == "__main__":
    main()
