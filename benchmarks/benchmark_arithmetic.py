"""Build and test the production integer core against frozen original source.

Requires a C++17 compiler, but neither PyTorch nor a GPU. --benchmark adds
single-process CPU measurements; omit it for correctness-only checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "benchmarks/reference/csrc/arithmetic_coding.cpp"


def extract_reference() -> str:
    """Replace Torch/Python adapters/checks; preserve every arithmetic step."""
    source = REFERENCE.read_text(encoding="utf-8")
    start = source.index("class OutCacheString")
    end = source.index("// ------------------------- Stateless")
    source = source[start:end]
    source = re.sub(r"TORCH_CHECK\(.*?\);", "", source, flags=re.S)
    source = source.replace("check_cdf_16(cdf);", "").replace("check_sym_1d(sym);", "")
    source = source.replace("const int64_t N = cdf.size(0);", "")
    source = source.replace(
        "void encode_chunk(const torch::Tensor& cdf, const torch::Tensor& sym)",
        "void encode_chunk(const uint16_t* cdf, const int16_t* sym, int64_t N)",
    )
    source = source.replace(
        "torch::Tensor decode_chunk(const torch::Tensor& cdf)",
        "std::vector<int16_t> decode_chunk(const uint16_t* cdf, int64_t N)",
    )
    source = source.replace(
        "reinterpret_cast<const cdf_t*>(cdf.data_ptr<int16_t>())", "cdf"
    )
    source = source.replace("sym.accessor<int16_t, 1>()", "sym")
    source = source.replace(
        "auto out = torch::empty({N}, torch::TensorOptions().dtype(torch::kInt16).device(torch::kCPU));",
        "std::vector<int16_t> out(static_cast<size_t>(N));",
    )
    source = source.replace("out.accessor<int16_t, 1>()", "out.data()")
    source = source.replace("py::bytes", "std::string")
    if "torch::" in source or "py::" in source:
        raise RuntimeError("Reference adapter needs updating; unconverted API remains")
    return (
        "#include <cassert>\n#include <cstdint>\n#include <string>\n#include <vector>\n"
        "namespace reference { using cdf_t = uint16_t;\n"
        "static constexpr int kLp=17, kMaxSym=15, kPrecision=16;\n"
        "static constexpr uint32_t kFull=65536U;\n" + source + "\n}\n"
    )


def geometry_histogram_samples(build: Path):
    """Actual sample occupancies with empirical CDFs; no neural predictions."""
    import numpy as np
    sys.path.insert(0, str(ROOT))
    from src.io import read_points

    def dilate(values):
        values = values.astype(np.uint64) & np.uint64(0x1fffff)
        for shift, mask in [(32, 0x1f00000000ffff), (16, 0x1f0000ff0000ff),
                            (8, 0x100f00f00f00f00f), (4, 0x10c30c30c30c30c3), (2, 0x1249249249249249)]:
            values = (values | (values << np.uint64(shift))) & np.uint64(mask)
        return values

    paths, metadata = [], []
    for dataset in ["Ford", "SemanticKITTI"]:
        for path in sorted((ROOT / "data" / (dataset + "Example")).glob("*.ply")):
            xyz = read_points(str(path))
            if dataset == "Ford":
                if xyz.min() < 0:
                    xyz = xyz + 131072
            else:
                xyz = xyz / 0.001 + 131072
            xyz = np.rint(xyz / 16).astype(np.int32)
            code = np.unique(dilate(xyz[:, 0]) | (dilate(xyz[:, 1]) << np.uint64(1)) |
                             (dilate(xyz[:, 2]) << np.uint64(2)))
            layers = []
            while True:
                parents, starts = np.unique(code >> np.uint64(3), return_index=True)
                occupancy = np.add.reduceat(np.uint32(1) << (code & np.uint64(7)).astype(np.uint32), starts)
                halves = [(occupancy & 15).astype(np.int16), (occupancy >> 4).astype(np.int16)]
                cdfs = []
                for symbols in halves:
                    histogram = np.bincount(symbols, minlength=16)
                    cdf = np.rint(np.concatenate(([0.0], np.cumsum(histogram) / histogram.sum())) * 65520)
                    cdf = (cdf.astype(np.uint32) + np.arange(17, dtype=np.uint32)).astype(np.uint16)
                    cdfs.append(np.tile(cdf, (len(symbols), 1)))
                layers.append((np.concatenate(cdfs), np.concatenate(halves)))
                if parents[-1] < 64:
                    break
                code = parents
            cdf = np.concatenate([layer[0] for layer in reversed(layers)])
            symbols = np.concatenate([layer[1] for layer in reversed(layers)])
            target = build / f"histogram_{dataset}_{path.stem}.bin"
            with target.open("wb") as stream:
                stream.write(np.array(len(symbols), dtype="<u8").tobytes())
                stream.write(cdf.astype("<u2").tobytes())
                stream.write(symbols.astype("<i2").tobytes())
            paths.append(str(target))
            metadata.append(dict(file=str(path.relative_to(ROOT)), quantization=16, input_points=len(xyz),
                                 layers=len(layers), symbols=len(symbols),
                                 cdf_model="per-layer empirical 16-way occupancy histogram; not neural CDF"))
    return paths, metadata


def run(output: Path | None = None, benchmark: bool = False, compiler: str | None = None,
        geometry_histograms: bool = False) -> dict:
    compiler = compiler or shutil.which("g++") or shutil.which("clang++") or shutil.which("cl")
    if not compiler:
        raise RuntimeError("A C++17 compiler (g++, clang++, or cl) is required")
    build_root = (ROOT / "reports/.arithmetic_tmp").resolve()
    build_root.relative_to(ROOT.resolve())
    build_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="elic-arithmetic-", dir=build_root) as temp:
        build = Path(temp)
        (build / "reference_core.h").write_text(extract_reference(), encoding="utf-8")
        executable = build / ("arithmetic_test.exe" if os.name == "nt" else "arithmetic_test")
        source = ROOT / "benchmarks/arithmetic_test.cpp"
        if Path(compiler).name.lower() in {"cl", "cl.exe"}:
            command = [compiler, "/nologo", "/O2", "/EHsc", "/std:c++17", f"/I{build}",
                       f"/I{ROOT / 'csrc'}", str(source), f"/Fe:{executable}", f"/Fo:{build / 'test.obj'}"]
        else:
            command = [compiler, "-O3", "-std=c++17", "-DNDEBUG", f"-I{build}",
                       f"-I{ROOT / 'csrc'}", str(source), "-o", str(executable)]
        subprocess.run(command, check=True, capture_output=True, text=True)
        paths, geometry_metadata = geometry_histogram_samples(build) if geometry_histograms else ([], [])
        completed = subprocess.run([str(executable)] + (["--benchmark"] if benchmark else []) + paths,
                                   check=True, capture_output=True, text=True)
        result = json.loads(completed.stdout)
    result["compiler"] = str(compiler)
    result["reference_sha256"] = hashlib.sha256(REFERENCE.read_bytes()).hexdigest()
    result["core_sha256"] = hashlib.sha256((ROOT / "csrc/arithmetic_core.h").read_bytes()).hexdigest()
    result["binding_sha256"] = hashlib.sha256((ROOT / "csrc/arithmetic_coding.cpp").read_bytes()).hexdigest()
    result["harness_sha256"] = hashlib.sha256((ROOT / "benchmarks/arithmetic_test.cpp").read_bytes()).hexdigest()
    result["measurement_scope"] = "standalone CPU integer arithmetic core; excludes Torch adapters, GPU, neural network and I/O"
    if geometry_metadata:
        result["geometry_histogram_inputs"] = geometry_metadata
        result["geometry_histogram_correctness_cases"] = len(geometry_metadata)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compiler")
    parser.add_argument("--geometry-histograms", action="store_true", help="also measure actual sample occupancy streams with empirical CDFs")
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.output, arguments.benchmark or arguments.geometry_histograms,
                         arguments.compiler, arguments.geometry_histograms), indent=2))
