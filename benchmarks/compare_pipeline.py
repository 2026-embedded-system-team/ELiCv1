"""CUDA acceptance test: frozen/optimized streams and geometry, both directions.

Uses frozen original algorithms in their own temporary directory. Only a main
guard is added to legacy drivers so Windows workers can import them safely.
Baseline needs original dependencies including Open3D; process logs are retained.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.io import read_points


def run(command, cwd, log):
    start = time.perf_counter()
    env = os.environ.copy()
    env.setdefault('TORCH_EXTENSIONS_DIR', str(log.parent/'.torch_extensions'))
    completed = subprocess.run([sys.executable, *map(str, command)], cwd=cwd,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    log.write_text(completed.stdout, encoding='utf-8')
    if completed.returncode:
        raise RuntimeError(f'process failed; see {log}')
    return time.perf_counter() - start


def sorted_rows(values):
    values = np.asarray(values, dtype=np.float32)
    return values[np.lexsort((values[:, 2], values[:, 1], values[:, 0]))]


def guard_legacy_driver(path):
    """Move only execution into main; leave imports available to spawn workers."""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    function = ast.FunctionDef(name='_legacy_main', args=ast.arguments(posonlyargs=[], args=[],
                              kwonlyargs=[], kw_defaults=[], defaults=[]), body=body,
                              decorator_list=[])
    guard = ast.parse("if __name__ == '__main__':\n    _legacy_main()\n").body[0]
    adapted = ast.Module(body=imports+[function, guard], type_ignores=[])
    path.write_text(ast.unparse(ast.fix_missing_locations(adapted))+'\n', encoding='utf-8')


def expected_geometry(path, posQ, prequantized):
    points = read_points(path)
    if prequantized:
        xyz = torch.tensor(points)
        if xyz.min().item() < 0:
            xyz += 131072
    else:
        xyz = torch.tensor(points / 0.001 + 131072)
    xyz = torch.unique(torch.round(xyz / posQ).int(), dim=0)
    xyz = xyz*posQ - 131072
    if not prequantized:
        xyz = xyz * .001
    return sorted_rows(xyz.float().numpy())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--posQ', nargs='+', type=int, choices=(4, 8, 16, 32, 64), default=[4, 8, 16, 32, 64])
    parser.add_argument('--datasets', nargs='+', choices=('Ford', 'SemanticKITTI'), default=['Ford', 'SemanticKITTI'])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--async_entropy', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT/'reports/pipeline_validation.json')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    missing = [m for m in ('torchsparse', 'open3d', 'ninja') if importlib.util.find_spec(m) is None]
    result = dict(status='unavailable', cuda=torch.cuda.is_available(), missing_modules=missing,
                  python=sys.version, torch=torch.__version__, posQ=args.posQ, repeats=args.repeats,
                  asynchronous=args.async_entropy, rows=[])
    if not result['cuda'] or missing:
        result['reason'] = 'Full trained-network compatibility and timing require CUDA, TorchSparse and baseline/JIT dependencies.'
        args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(result['reason'], flush=True)
        raise SystemExit(2)
    result['gpu'] = torch.cuda.get_device_name()
    manifest = json.loads((ROOT/'benchmarks/reference/manifest.json').read_text(encoding='utf-8'))
    for relative, expected in manifest['files'].items():
        actual = hashlib.sha256((ROOT/'benchmarks/reference'/relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f'baseline has changed: {relative}')
    logs = args.output.parent/'pipeline_logs'
    logs.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='pipeline-', dir=args.output.parent) as temp:
        work = Path(temp)
        baseline = work/'baseline'
        shutil.copytree(ROOT/'benchmarks/reference', baseline, ignore=shutil.ignore_patterns('__pycache__'))
        for driver in ('compress.py', 'decompress.py'):
            guard_legacy_driver(baseline/driver)
        result['baseline_adapter'] = 'Execution-only main guards added to temporary drivers for Windows spawn; original algorithms and weights retained.'
        optimized = work/'optimized'
        optimized.mkdir()
        for dataset in args.datasets:
            source = ROOT/'data'/f'{dataset}Example'
            files = sorted(source.glob('*.ply'))
            if not files:
                raise RuntimeError(f'no input samples in {source}')
            pre = dataset == 'Ford'
            options = ['--ckpt', ROOT/'model'/dataset/'ELiCv1_K5.pt']
            if pre:
                options += ['--is_data_pre_quantized', 'True']
            tag = 'Ford' if pre else 'KITTI'
            for posQ in args.posQ:
                csv = f'{tag}_K5_posQ{posQ}.csv'
                timings = {key: [] for key in ('baseline_enc', 'optimized_enc', 'baseline_dec', 'optimized_dec',
                                              'baseline_process', 'optimized_process')}
                roots = {'baseline': baseline, 'optimized': optimized}
                outputs = {}
                for repeat in range(args.repeats + 1):
                    # The first full run warms compilation, CUDA and disk caches.
                    order = ('baseline', 'optimized') if repeat % 2 == 0 else ('optimized', 'baseline')
                    for implementation in order:
                        cwd = roots[implementation]
                        script_root = baseline if implementation == 'baseline' else ROOT
                        # Legacy writer appends if its shell 'rm' fails on
                        # Windows. Fresh files per repetition avoid stale data.
                        out = cwd/f'{dataset}_q{posQ}_r{repeat}'
                        out.mkdir(exist_ok=True)
                        outputs[implementation] = out
                        enc_cmd = [script_root/'compress.py', '--input_glob', source/'*.ply',
                                   '--output_folder', out, '--posQ', posQ, *options]
                        if implementation == 'optimized' and args.async_entropy:
                            enc_cmd += ['--async_entropy', 'True']
                        label = f'{dataset}_q{posQ}_{implementation}_r{repeat}'
                        total = run(enc_cmd, cwd, logs/(label+'_enc.log'))
                        total += run([script_root/'decompress.py', '--input_glob', out/'*.bin',
                                      '--output_folder', out, *options], cwd, logs/(label+'_dec.log'))
                        frame = pd.read_csv(cwd/csv)
                        if repeat:
                            timings[implementation+'_enc'].append(float(frame.enc_time.mean()))
                            timings[implementation+'_dec'].append(float(frame.dec_time.mean()))
                            timings[implementation+'_process'].append(total)
                    for source_path in files:
                        stem = source_path.stem
                        if (outputs['baseline']/(stem+'.bin')).read_bytes() != (outputs['optimized']/(stem+'.bin')).read_bytes():
                            raise AssertionError(f'compressed bytes differ: {dataset}/{stem}, posQ={posQ}')
                        expected = expected_geometry(source_path, posQ, pre)
                        for name in outputs:
                            actual = sorted_rows(read_points(outputs[name]/(stem+'.ply')))
                            if not np.array_equal(actual, expected):
                                raise AssertionError(f'quantized geometry mismatch: {name}/{stem}, posQ={posQ}')
                # Cross-version decoding proves original->new and new->original.
                for decoder_name, encoder_name in (('optimized', 'baseline'), ('baseline', 'optimized')):
                    destination = roots[decoder_name]/f'{dataset}_cross_q{posQ}'
                    destination.mkdir(exist_ok=True)
                    script_root = ROOT if decoder_name == 'optimized' else baseline
                    cmd = [script_root/'decompress.py', '--input_glob', outputs[encoder_name]/'*.bin',
                           '--output_folder', destination, *options]
                    if decoder_name == 'optimized':
                        cmd += ['--csv', destination/'cross.csv']
                    run(cmd, roots[decoder_name], logs/f'{dataset}_q{posQ}_{encoder_name}_to_{decoder_name}.log')
                    for source_path in files:
                        actual = sorted_rows(read_points(destination/(source_path.stem+'.ply')))
                        if not np.array_equal(actual, expected_geometry(source_path, posQ, pre)):
                            raise AssertionError(f'cross-decoding failed: {encoder_name}->{decoder_name}')
                row = dict(dataset=dataset, posQ=posQ, clouds=len(files), byte_identical=True,
                           exact_quantized_geometry=True, cross_decode=True, samples_seconds=timings)
                for operation in ('enc', 'dec', 'process'):
                    old = statistics.median(timings['baseline_'+operation])
                    new = statistics.median(timings['optimized_'+operation])
                    row[operation] = dict(baseline_seconds=old, optimized_seconds=new, speedup=old/new)
                result['rows'].append(row)
                args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
                print(f'{dataset} posQ={posQ}: bytes/geometry/cross-decode passed; '
                      f'enc {row["enc"]["speedup"]:.3f}x, dec {row["dec"]["speedup"]:.3f}x', flush=True)
    result['status'] = 'passed'
    result['timing_note'] = ('Codec CSV excludes model/JIT/input reading/output writing; process wall time includes these. '
                             'Baseline encode uses CUDA events; optimized uses synchronized perf_counter. '
                             'Each subprocess performs original-style neural warmup. No frames are discarded.')
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
