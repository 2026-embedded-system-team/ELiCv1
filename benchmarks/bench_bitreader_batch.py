"""Check/timing word-refill reader against a supplied production-core snapshot."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--geometry-histograms', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(ROOT) and 'reports' in {part.casefold() for part in output.relative_to(ROOT).parts}:
        parser.error('write experiment results outside reports/')
    compiler = shutil.which('g++') or shutil.which('clang++')
    if not compiler:
        raise RuntimeError('g++ or clang++ is required')
    before = args.before.resolve()
    sources = {
        'before_core': before,
        'after_core': ROOT / 'csrc/arithmetic_core.h',
        'cpp_harness': ROOT / 'benchmarks/bitreader_batch_test.cpp',
        'python_harness': Path(__file__).resolve(),
        'geometry_generator': ROOT / 'benchmarks/benchmark_arithmetic.py',
        'point_cloud_io': ROOT / 'src/io.py',
    }
    source_hashes = {name: {'path': str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                            'sha256': sha256(path)} for name, path in sources.items()}
    flags = ['-O3', '-std=c++17', '-DNDEBUG']
    compiler_version = subprocess.run([compiler, '--version'], check=True, capture_output=True, text=True).stdout.splitlines()[0]
    build = output.parent / '.bitreader_build'
    build.mkdir(parents=True, exist_ok=True)
    (build / 'before_arithmetic_core.h').write_bytes(before.read_bytes())
    executable = build / ('bitreader_batch_test.exe' if os.name == 'nt' else 'bitreader_batch_test')
    compile_command = [compiler, *flags, f'-I{ROOT / "csrc"}', f'-I{build}',
                       str(ROOT / 'benchmarks/bitreader_batch_test.cpp'), '-o', str(executable)]
    subprocess.run(compile_command, check=True)
    geometry_paths, geometry_metadata = [], []
    if args.geometry_histograms:
        from benchmarks.benchmark_arithmetic import geometry_histogram_samples
        geometry_paths, geometry_metadata = geometry_histogram_samples(build)
        for sample_file, metadata in zip(geometry_paths, geometry_metadata):
            sample_path = Path(sample_file)
            source_path = ROOT / metadata['file']
            metadata.update(source_sha256=sha256(source_path), source_bytes=source_path.stat().st_size,
                            sample_sha256=sha256(sample_path), sample_bytes=sample_path.stat().st_size,
                            cdf_bytes=metadata['symbols'] * 34, symbol_bytes=metadata['symbols'] * 2)
    completed = subprocess.run([str(executable)] + (['--benchmark'] if args.benchmark or args.geometry_histograms else []) + geometry_paths,
                               check=True, capture_output=True, text=True)
    result = json.loads(completed.stdout)
    result.update(compiler=compiler, baseline=str(before),
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  environment=dict(python=sys.version, python_executable=sys.executable,
                                   platform=platform.platform(), machine=platform.machine(),
                                   processor=os.environ.get('PROCESSOR_IDENTIFIER') or platform.processor(),
                                   logical_cpus=os.cpu_count(), device='cpu', decoder_threads=1),
                  compiler_version=compiler_version, compiler_flags=flags, compile_command=compile_command,
                  source_hashes=source_hashes,
                  timing_config=dict(repeats=15, warmups=3, order='alternating before/after pairs',
                                     aggregation='median', clock='std::chrono::steady_clock', unit='milliseconds',
                                     includes=['decoder reset and bitstream copy', 'complete decode_chunk'],
                                     excludes=['CDF/input generation', 'correctness checks', 'compilation',
                                               'output allocation (preallocated equally for both decoders)']),
                  random_seed=20261020,
                  baseline_sha256=source_hashes['before_core']['sha256'],
                  optimized_sha256=source_hashes['after_core']['sha256'],
                  scope='CPU integer decoder only; same CDFs, outputs and reset copies; no neural/GPU timing')
    if geometry_metadata:
        result['geometry_histogram_inputs'] = geometry_metadata
        result['geometry_correctness_cases'] = len(geometry_metadata)
        geometry_rows = [row for row in result['rows'] if row['case'].startswith('histogram_')]
        before_sum = sum(row['before_ms'] for row in geometry_rows)
        after_sum = sum(row['after_ms'] for row in geometry_rows)
        result['geometry_aggregate'] = dict(before_sum_medians_ms=before_sum, after_sum_medians_ms=after_sum,
                                            speedup=before_sum / after_sum,
                                            time_reduction_percent=(1 - after_sum / before_sum) * 100,
                                            aggregation='sum of per-cloud paired medians; not mean speedup')
        for row, metadata in zip(geometry_rows, geometry_metadata):
            row['input_sample_sha256'] = metadata['sample_sha256']
    for name, path in sources.items():
        if sha256(path) != source_hashes[name]['sha256']:
            raise RuntimeError(f'Source changed during benchmark; rerun after edits finish: {path}')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
