"""Paired module timings. Neural/TorchSparse end-to-end requires compare_pipeline.py."""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import morton, op


def load_reference(name):
    spec = importlib.util.spec_from_file_location('reference_' + name, ROOT / 'benchmarks/reference/src' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def paired(name, before, after, repeats, device, inner=1):
    def sample(fn):
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        start = time.perf_counter_ns()
        for _ in range(inner):
            result = fn()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        elapsed = (time.perf_counter_ns()-start)/1e6/inner
        del result
        return elapsed
    for _ in range(3):
        before()
        after()
    a, b = [], []
    for repeat in range(repeats):
        order = ((before, a), (after, b)) if repeat % 2 == 0 else ((after, b), (before, a))
        for fn, target in order:
            target.append(sample(fn))
    old, new = statistics.median(a), statistics.median(b)
    return dict(module=name, baseline_ms=old, optimized_ms=new, speedup=old/new,
                baseline_samples_ms=a, optimized_samples_ms=b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu', choices=('cpu', 'cuda'))
    parser.add_argument('--threads', default=1, type=int)
    parser.add_argument('--repeats', default=11, type=int)
    parser.add_argument('--points', default=100000, type=int)
    parser.add_argument('--output', default='reports/modules_cpu.json')
    args = parser.parse_args()
    if min(args.threads, args.repeats, args.points) < 1:
        parser.error('--threads, --repeats, and --points must be positive')
    torch.set_num_threads(args.threads)
    torch.manual_seed(11)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable')
    ref = load_reference('morton')
    ref_op = load_reference('op')
    # Reference used CUDA-only global lookup tables. Relocate those unchanged
    # tables to the test device; its arithmetic and allocations remain intact.
    ref._get_occ_bits._occ_bits = ((torch.arange(256, device=device)[:, None] >>
                                  torch.arange(8, device=device)[None, :]) & 1).bool()
    ref._get_expand_coords_base._table = torch.tensor(
        [[i & 1, (i >> 1) & 1, i >> 2] for i in range(8)], dtype=torch.int32, device=device)
    coords = torch.randint(0, 16384, (args.points, 4), device=device, dtype=torch.int32)
    coords[:, 0] = 0
    xyz, codes = morton.morton3_sort(coords)
    parents, _, occ = morton.down_once(xyz, codes)
    features = torch.randn((len(parents), 32), device=device)
    cdf = torch.rand((args.points, 16), device=device).softmax(dim=-1).cumsum(dim=-1)
    cdf = torch.cat((cdf[:, :1]*0, cdf), dim=-1)

    def old_hierarchy():
        points, code = ref.morton3_sort(coords)
        levels = []
        while True:
            points, code, occupancy = ref.down_once(points, code)
            levels.append((points.clone(), occupancy.clone()))
            if int(points[:, 1:].max().item()).bit_length() <= 2:
                break
        return levels[::-1]

    cases = [
        ('morton_sort', lambda: ref.morton3_sort(coords), lambda: morton.morton3_sort(coords)),
        ('down_once', lambda: ref.down_once(xyz, codes), lambda: morton.down_once(xyz, codes)),
        ('upscale_feature', lambda: ref.upscale_feature(occ, features), lambda: morton.upscale_feature(occ, features)),
        ('upscale_coordinate', lambda: ref.upscale_coordinate(parents, occ), lambda: morton.upscale_coordinate(parents, occ)),
        ('upscale_coordinate_feature', lambda: ref.upscale_coordinate_feature(parents, occ, features),
         lambda: morton.upscale_coordinate_feature(parents, occ, features)),
        ('hierarchy', old_hierarchy, lambda: morton.build_hierarchy(coords)),
        ('cdf_normalization', lambda: ref_op._convert_to_int_and_normalize(cdf, True),
         lambda: op._convert_to_int_and_normalize(cdf, True)),
        ('cdf_normalization_64', lambda: ref_op._convert_to_int_and_normalize(cdf[:64], True),
         lambda: op._convert_to_int_and_normalize(cdf[:64], True)),
        ('cdf_normalization_4096', lambda: ref_op._convert_to_int_and_normalize(cdf[:4096], True),
         lambda: op._convert_to_int_and_normalize(cdf[:4096], True)),
    ]
    rows = []
    with torch.inference_mode():
        for name, before, after in cases:
            row = paired(name, before, after, args.repeats, device)
            rows.append(row)
            print(f'{name}: {row["baseline_ms"]:.3f} -> {row["optimized_ms"]:.3f} ms, {row["speedup"]:.3f}x', flush=True)
    output = dict(environment=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
                                  platform=platform.platform(), processor=os.environ.get('PROCESSOR_IDENTIFIER', platform.processor()),
                                  device=str(device), threads=args.threads), points=args.points,
                  repeats=args.repeats, seed=11, timing='alternating paired median; 3 warmups; wall clock; CUDA synchronized',
                  coordinate_sha256=hashlib.sha256(coords.cpu().numpy().tobytes()).hexdigest(), rows=rows)
    path = ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
