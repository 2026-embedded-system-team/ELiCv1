"""Benchmark offline BoE center-training histogram collection, not inference.

Extracts the original loop from frozen BoE_cluster.py without importing its
TorchSparse, matplotlib or sklearn dependencies. Uses PCDataset's float32,
posQ=4, non-augmented quantization on all four complete example clouds.
"""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import boe, io, morton


def _reference_module(name):
    spec = importlib.util.spec_from_file_location(
        "reference_boe_" + name, ROOT / "benchmarks/reference/src" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_frozen_functions():
    """Compile unchanged original collection/statistics/selection AST blocks."""
    path = ROOT / "benchmarks/reference/BoE_cluster.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    bitdepth_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "compute_bitdepth")
    main_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    collection_loop = next(n for n in main_fn.body if isinstance(n, ast.While))
    frame_loop = next(n for n in collection_loop.body if isinstance(n, ast.For))
    # Exclude only the data-loader input fetch and its outer break. The original
    # hierarchy, sorting, nibble operations, counts, labels and truncation stay.
    frame_statements = copy.deepcopy(frame_loop.body[1:-1])
    collect_fn = ast.parse("""
def frozen_collect(coords, max_samples=64):
    device = coords.device
    x = SimpleNamespace(coords=coords)
    cnt_samples = 0
    x_Hs = torch.zeros((max_samples, 32), dtype=torch.float32, device=device)
    x_Bs = torch.zeros((max_samples, 1), dtype=torch.int32, device=device)
    return x_Hs[:cnt_samples], x_Bs[:cnt_samples]
""").body[0]
    collect_fn.body[-1:-1] = frame_statements
    depth_loop = next(n for n in frame_statements if isinstance(n, ast.For))
    # Keep original coordinate+feature sorting and all histogram operations.
    first_store = next(i for i, n in enumerate(depth_loop.body)
                       if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Subscript)
                       and isinstance(n.targets[0].value, ast.Name)
                       and n.targets[0].value.id == "x_Hs")
    histogram_fn = ast.parse("""
def frozen_histogram(x_C, x_O):
    return x_H
""").body[0]
    histogram_fn.body[0:0] = copy.deepcopy(depth_loop.body[1:first_store])
    model_tree = ast.parse((ROOT / "benchmarks/reference/model.py").read_text(encoding="utf-8"))
    model_class = next(n for n in model_tree.body if isinstance(n, ast.ClassDef) and n.name == "ELiCv1")
    selection_fn = copy.deepcopy(next(n for n in model_class.body
                                     if isinstance(n, ast.FunctionDef) and n.name == "select_encoder_index"))
    selection_fn.name = "frozen_select_encoder_index"
    module_tree = ast.Module(body=[copy.deepcopy(bitdepth_fn), collect_fn,
                                  histogram_fn, selection_fn], type_ignores=[])
    environment = dict(torch=torch, math=math, SimpleNamespace=SimpleNamespace,
                       mt=_reference_module("morton"), op=_reference_module("op"))
    exec(compile(ast.fix_missing_locations(module_tree), str(path), "exec"), environment)
    return (environment["frozen_collect"], environment["frozen_histogram"],
            environment["frozen_select_encoder_index"])


def main():
    # Importing the helper is also supported by the unit tests; timing runs
    # happen only under the explicit CLI main guard.
    from bench_morton import time_pair
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", default=1, type=int)
    parser.add_argument("--warmup", default=2, type=int)
    parser.add_argument("--repeats", default=7, type=int)
    parser.add_argument("--posq", default=4, type=int)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/boe_cpu.json")
    args = parser.parse_args()
    if min(args.threads, args.repeats, args.posq) < 1 or args.warmup < 0:
        parser.error("threads, repeats, posq must be positive and warmup nonnegative")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    frozen_collect, frozen_histogram, _ = load_frozen_functions()
    report = dict(scope="offline BoE center-training histogram preprocessing; not inference selection or KMeans/TSNE",
                  device=str(device), threads=args.threads, warmup=args.warmup, repeats=args.repeats,
                  torch_version=torch.__version__, python_version=platform.python_version(),
                  platform=platform.platform(), posq=args.posq,
                  quantization="PCDataset float32; no augmentation; Ford prequantized; KITTI /0.001; unconditional +131072; round /posq",
                  baseline="AST-extracted unchanged blocks from benchmarks/reference/BoE_cluster.py",
                  timer="perf_counter_ns; alternating pair order; median; CUDA synchronize when applicable", samples=[])
    with torch.inference_mode():
        for path in sorted((ROOT / "data").glob("*Example/*.ply")):
            points = torch.tensor(io.read_points(path), dtype=torch.float32)
            prequantized = "FordExample" in path.parts
            if not prequantized:
                points = points / .001
            xyz = torch.round((points + 131072) / args.posq).int()
            coords = torch.cat((xyz[:, :1] * 0, xyz), dim=1).to(device)
            hierarchy = morton.build_hierarchy(coords, stop_bitdepth=7)
            original_h, original_b = frozen_collect(coords)
            new_h, new_b = boe.collect_histograms(coords)
            assert torch.equal(original_h, new_h) and torch.equal(original_b, new_b)
            stages = {
                "boe_offline_sort_histograms": (
                    lambda: torch.stack([frozen_histogram(c, o) for c, o in hierarchy]),
                    lambda: torch.stack([boe.occupancy_histogram(o) for _, o in hierarchy])),
                "boe_offline_collect": (lambda: frozen_collect(coords),
                                        lambda: boe.collect_histograms(coords)),
            }
            sample = dict(file=str(path.relative_to(ROOT)), prequantized=prequantized,
                          source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          quantized_coords_sha256=hashlib.sha256(coords.cpu().numpy().tobytes()).hexdigest(),
                          input_points=len(coords), levels=len(hierarchy), labels=new_b.view(-1).tolist(),
                          histogram_exact=True, bitdepth_exact=True, stages={})
            for name, (before, after) in stages.items():
                result = time_pair(before, after, device, args.warmup, args.repeats)
                sample["stages"][name] = result
                print(f"{sample['file']} {name}: {result['baseline_ms']:.3f} -> "
                      f"{result['optimized_ms']:.3f} ms ({result['speedup']:.2f}x)", flush=True)
            report["samples"].append(sample)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
