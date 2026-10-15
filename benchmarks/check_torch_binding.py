"""Compile the actual Torch wrapper without Ninja, then run CPU API checks.

Run using a Python environment with PyTorch and setuptools already installed.
On Windows an installed Visual Studio C++ toolchain is used automatically.
No packages are installed and no CUDA/TorchSparse is required.
"""
from __future__ import annotations

import argparse
import glob
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def verify(output=None):
    build_root = (ROOT / "reports/.arithmetic_tmp").resolve()
    build_root.relative_to(ROOT.resolve())
    build_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="torch-binding-", dir=build_root) as temp:
        build = Path(temp)
        setup_script = build / "setup.py"
        flags = ["/O2", "/std:c++17"] if os.name == "nt" else ["-O3", "-std=c++17"]
        setup_script.write_text(
            "from setuptools import setup\n"
            "from torch.utils.cpp_extension import CppExtension, BuildExtension\n"
            f"setup(name='elic_ac_binding_check', ext_modules=[CppExtension('elic_ac_binding_check', "
            f"sources={[str(ROOT / 'csrc/arithmetic_coding.cpp')]!r}, extra_compile_args={flags!r})], "
            "cmdclass={'build_ext':BuildExtension.with_options(use_ninja=False)})\n", encoding="utf-8")
        command = [sys.executable, str(setup_script), "build_ext", "--inplace"]
        environment = os.environ.copy()
        environment["MAX_JOBS"] = "1"
        if os.name == "nt" and not shutil.which("cl"):
            locations = glob.glob(os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"),
                                               "Microsoft Visual Studio", "*", "*", "VC", "Auxiliary", "Build", "vcvars64.bat"))
            if not locations:
                raise RuntimeError("Visual Studio C++ compiler was not found")
            batch = build / "build.cmd"
            batch.write_text(
                f'@echo off\ncall "{sorted(locations)[-1]}" >nul\n'
                'if errorlevel 1 exit /b %errorlevel%\n'
                f'"{sys.executable}" "{setup_script}" build_ext --inplace\n', encoding="utf-8")
            command = ["cmd.exe", "/d", "/c", str(batch)]
            environment["DISTUTILS_USE_SDK"] = "1"
            environment["MSSdk"] = "1"
        completed = subprocess.run(command, cwd=build, env=environment, capture_output=True, text=True,
                                   encoding="utf-8", errors="replace")
        build_log = ROOT / "reports/arithmetic_torch_build.log"
        build_log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
        if completed.returncode:
            raise RuntimeError(f"Torch extension build failed; see {build_log}\n" + (completed.stdout + completed.stderr)[-3000:])
        checked = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--smoke-build", str(build)],
                                 cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if checked.returncode:
            raise RuntimeError("Torch binding smoke test failed: " + checked.stdout + checked.stderr)
        result = json.loads(checked.stdout)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _smoke_binding(build):
    import torch
    build = Path(build)
    build_log = ROOT / "reports/arithmetic_torch_build.log"
    sys.path.insert(0, str(build))
    module = importlib.import_module("elic_ac_binding_check")
    generator = torch.Generator().manual_seed(11)
    probabilities = torch.rand((2048, 16), generator=generator)
    probabilities /= probabilities.sum(1, keepdim=True)
    cdf = torch.cat((torch.zeros(2048, 1), probabilities.cumsum(-1)), -1)
    cdf = (cdf * 65520).round().to(torch.int16) + torch.arange(17, dtype=torch.int16)
    symbols = torch.randint(0, 16, (2048,), generator=generator, dtype=torch.int16)
    one_shot = module.encode_cdf_16(cdf, symbols)
    encoder = module.ArithmeticEncoder16()
    encoder.encode_chunk(cdf[:333], symbols[:333])
    encoder.encode_chunk(cdf[333:333], symbols[333:333])
    encoder.encode_chunk(cdf[333:], symbols[333:])
    assert encoder.finish() == one_shot
    decoder = module.ArithmeticDecoder16()
    for _ in range(2):
        decoder.reset(one_shot)
        decoded = torch.cat((decoder.decode_chunk(cdf[:701]), decoder.decode_chunk(cdf[701:])))
        assert torch.equal(decoded, symbols)
    assert torch.equal(module.decode_cdf_16(cdf, one_shot), symbols)
    encoder.encode_chunk(cdf, symbols)
    assert encoder.finish() == one_shot
    rejected = 0
    for bad_cdf, bad_symbols in [(cdf[:, :16].contiguous(), symbols), (cdf.float(), symbols),
                                 (cdf, symbols.long()), (cdf, symbols[:-1]),
                                 (cdf[:1], torch.tensor([16], dtype=torch.int16))]:
        try:
            module.encode_cdf_16(bad_cdf, bad_symbols)
        except (RuntimeError, ValueError, IndexError):
            rejected += 1
    assert rejected == 5
    result = {"status": "passed", "torch_version": torch.__version__, "cuda_available": torch.cuda.is_available(),
              "checks": ["actual Torch extension compilation", "chunk byte equality", "empty chunk", "two-stage decode",
                         "stateless helpers", "encoder reuse", "decoder reset", "five invalid input cases"],
              "build_log": str(build_log.relative_to(ROOT))}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--smoke-build", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        result = _smoke_binding(args.smoke_build) if args.smoke_build else verify(args.output)
    except (RuntimeError, OSError) as error:
        result = {"status": "unavailable", "reason": str(error),
                  "build_log": "reports/arithmetic_torch_build.log"}
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        raise SystemExit(1)
    print(json.dumps(result, indent=2))
