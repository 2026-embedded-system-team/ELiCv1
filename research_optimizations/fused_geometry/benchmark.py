"""Isolated current-vs-fused CPU proof; production and reports are not edited."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "portable_bench"))
from elic_bench.backend import Backend
from elic_bench.config import normalize_config
from elic_bench.context import Context, exact_equal
from elic_bench.native import _compiler_command
from elic_bench.results import summarize, write_outputs
from elic_bench.suites import geometry_inputs, load_modules


def load_current(name):
    path = ROOT / "src/morton.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FusedGeometry:
    def __init__(self, library):
        self.library = library
        pointer, integer = ctypes.c_void_p, ctypes.c_int64
        for suffix in ("i32", "i64"):
            getattr(library, "elic_morton_" + suffix).argtypes = [pointer, integer, integer, integer, pointer]
            getattr(library, "elic_morton_" + suffix).restype = None
            getattr(library, "elic_count_" + suffix).argtypes = [pointer, pointer, integer, integer, integer]
            getattr(library, "elic_count_" + suffix).restype = integer
            getattr(library, "elic_down_" + suffix).argtypes = [pointer, pointer, integer, integer, integer, integer, pointer, pointer, pointer]
            getattr(library, "elic_down_" + suffix).restype = None

    def suffix(self, tensor):
        if tensor.device.type != "cpu" or tensor.dtype not in (torch.int32, torch.int64):
            raise TypeError("Research fused backend accepts CPU int32/int64 tensors only")
        return "i32" if tensor.dtype == torch.int32 else "i64"

    def code(self, xyz):
        suffix = self.suffix(xyz)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("XYZ must have shape (N,3)")
        out = torch.empty(len(xyz), dtype=torch.int64)
        getattr(self.library, "elic_morton_" + suffix)(xyz.data_ptr(), len(xyz), *xyz.stride(), out.data_ptr())
        return out

    def down_once(self, coords, code):
        suffix = self.suffix(coords)
        if coords.ndim != 2 or coords.shape[1] != 4 or code.dtype != torch.int64 or code.device.type != "cpu" or code.ndim != 1 or len(code) != len(coords):
            raise ValueError("Expected CPU [batch,x,y,z] coordinates and corresponding int64 Morton codes")
        count = getattr(self.library, "elic_count_" + suffix)(coords.data_ptr(), code.data_ptr(), len(coords), coords.stride(0), code.stride(0))
        parents = torch.empty((count, 4), dtype=torch.int32)
        parent_codes = torch.empty(count, dtype=torch.int64)
        occupancy = torch.empty(count, dtype=torch.int32)
        getattr(self.library, "elic_down_" + suffix)(coords.data_ptr(), code.data_ptr(), len(coords), *coords.stride(), code.stride(0), parents.data_ptr(), parent_codes.data_ptr(), occupancy.data_ptr())
        return parents, parent_codes, occupancy


@contextmanager
def compiled(build_root, compiler):
    prefix = _compiler_command(compiler)
    msvc = any(Path(token).name.lower() in ("cl", "cl.exe") for token in prefix)
    build_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fused-geometry-", dir=build_root) as directory:
        build = Path(directory)
        binary = build / ("fused.dll" if os.name == "nt" else "fused.dylib" if sys.platform == "darwin" else "fused.so")
        source = Path(__file__).with_name("fused_geometry.cpp")
        if msvc:
            command = prefix + ["/nologo", "/O2", "/EHsc", "/std:c++17", "/LD", str(source), f"/Fo{build / 'fused.obj'}", "/link", f"/OUT:{binary}", f"/IMPLIB:{build / 'fused.lib'}"]
        else:
            command = prefix + ["-O3", "-std=c++17", "-dynamiclib" if sys.platform == "darwin" else "-shared"]
            if os.name != "nt":
                command += ["-fPIC"]
            command += [str(source), "-o", str(binary)]
        subprocess.run(command, cwd=build, check=True, capture_output=True)
        directories, library = [], None
        try:
            if os.name == "nt":
                for path in [str(Path(prefix[0]).parent), *os.environ.get("PATH", "").split(os.pathsep)]:
                    if path and Path(path).is_dir():
                        directories.append(os.add_dll_directory(path))
            library = ctypes.CDLL(str(binary))
            yield FusedGeometry(library), dict(compiler_command=command,
                prototype_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                current_morton_sha256=hashlib.sha256((ROOT / "src/morton.py").read_bytes()).hexdigest(),
                isa="portable C++17; no -march=native, BMI2, CUDA or fast-math flags")
        finally:
            if library:
                import _ctypes
                (_ctypes.FreeLibrary if os.name == "nt" else _ctypes.dlclose)(library._handle)
                library._handle = 0
            for handle in reversed(directories):
                handle.close()


def edge_checks(before, after):
    generator = torch.Generator().manual_seed(73)
    checks = 0
    for dtype in (torch.int32, torch.int64):
        for n in (0, 1, 8, 257, 4097):
            # Row and column strides both differ from contiguous codec input.
            backing = torch.randint(0, 1 << 21, (n * 2, 8), generator=generator, dtype=dtype)
            coords = backing[::2, ::2]
            coords[:, 0] = 0
            original = coords.clone()
            exact_equal(before.morton3_code(coords[:, 1:]), after.morton3_code(coords[:, 1:]))
            ordered, codes = before.morton3_sort(coords)
            exact_equal(before.down_once(ordered, codes), after.down_once(ordered, codes))
            exact_equal(before.build_hierarchy(coords), after.build_hierarchy(coords))
            exact_equal(coords, original)
            checks += 4
        coords = torch.tensor([[7, 3, 1, 0], [-2, 0, 0, 0], [7, 0, 0, 0], [-2, 3, 1, 0], [7, 3, 1, 0], [99, 0, 0, 0]], dtype=dtype)
        exact_equal(before.build_hierarchy(coords, batch_mode="multiple"), after.build_hierarchy(coords, batch_mode="multiple"))
        xyz = torch.tensor([[0, 0, 0], [(1 << 21)-1]*3, [1 << 20, 0, 0], [0, 1 << 20, 0], [0, 0, 1 << 20]], dtype=dtype)
        expected = torch.tensor([sum(((v >> bit) & 1) << (3*bit+axis) for axis, v in enumerate(row) for bit in range(21)) for row in xyz.tolist()])
        exact_equal(expected, after.morton3_code(xyz))
        checks += 2
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("results"))
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--compiler")
    parser.add_argument("--correctness-only", action="store_true")
    args = parser.parse_args()
    if not __debug__:
        parser.error("Correctness checks require Python without -O")
    config = normalize_config(dict(repo_root=str(ROOT), output_dir=str(args.output_dir.resolve()),
                                  device="cpu", repeats=args.repeats, warmups=args.warmups,
                                  compiler=args.compiler, correctness_only=args.correctness_only))
    torch.set_num_threads(config["threads"])
    ctx = Context(config, Backend("cpu", torch))
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    before, after = load_current("fused_before"), load_current("fused_after")
    with compiled(ctx.output_dir / "build", args.compiler) as (fused, metadata):
        after.morton3_code = fused.code
        after.down_once = fused.down_once
        checks = edge_checks(before, after)
        _, modules = load_modules(ROOT)
        for label, coords, details in geometry_inputs(ctx, modules.io):
            ordered, codes = before.morton3_sort(coords)
            stages = {"morton3_code": (lambda: before.morton3_code(coords[:, 1:]), lambda: after.morton3_code(coords[:, 1:])),
                      "morton3_sort": (lambda: before.morton3_sort(coords), lambda: after.morton3_sort(coords)),
                      "down_once": (lambda: before.down_once(ordered, codes), lambda: after.down_once(ordered, codes)),
                      "build_hierarchy": (lambda: before.build_hierarchy(coords), lambda: after.build_hierarchy(coords))}
            for name, (old, new) in stages.items():
                row = ctx.record_case(f"{label}.{name}", name, old, new,
                    details={**details, "uses_cuda": False, "baseline": "current optimized source before fused prototype", "timing_scope": "CPU operator with Python/ctypes bridge and output allocation; excludes one-time compilation"})
                if row["timing"]:
                    timing = row["timing"]
                    print(f"{label:32s} {name:18s} {timing['baseline']['median_ms']:.4f} -> {timing['candidate']['median_ms']:.4f} ms ({timing['speedup']:.3f}x)", flush=True)
                else:
                    print(row["case_id"], row["status"], row.get("reason", ""), flush=True)
        report = dict(schema_version="1.0", baseline="current optimized implementation; not frozen original", native_build=metadata,
                      config=config, edge_checks=checks, cases=ctx.records, summary=summarize(ctx.records))
        for name in ("morton3_code", "morton3_sort", "down_once", "build_hierarchy"):
            selected = [row for row in ctx.records if row["suite"] == name and row["timing"]]
            if selected:
                baseline = sum(row["timing"]["baseline"]["median_ms"] for row in selected)
                candidate = sum(row["timing"]["candidate"]["median_ms"] for row in selected)
                report.setdefault("aggregate", {})[name] = dict(baseline_sum_medians_ms=baseline, candidate_sum_medians_ms=candidate, speedup=baseline/candidate, time_reduction_percent=(1-candidate/baseline)*100)
        write_outputs(report, ctx.output_dir)
        print(json.dumps(report.get("aggregate", {}), indent=2))
        raise SystemExit(report["summary"]["exit_code"])


if __name__ == "__main__":
    main()
