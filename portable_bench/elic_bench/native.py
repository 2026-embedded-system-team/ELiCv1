"""Portable, isolated C++ builds for the arithmetic benchmark.

All generated files live under the caller's build directory.  No project
report, cache, source, compiler configuration, or Python import path is changed.
"""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
import hashlib
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Iterator


class NativeUnavailable(RuntimeError):
    """A required compiler or runtime library is unavailable on this machine."""


class NativeBuildError(RuntimeError):
    """Sources could not be adapted or compiled; this is a failed test setup."""


def extract_reference(path: Path) -> str:
    """Remove only Torch adapters from the frozen original integer coder.

    The transformations follow benchmarks/benchmark_arithmetic.py, with an
    explicit source path instead of that module's repository/report globals.
    Arithmetic operations and their ordering are preserved.
    """
    source = path.read_text(encoding="utf-8")
    try:
        source = source[source.index("class OutCacheString"):
                        source.index("// ------------------------- Stateless")]
    except ValueError as error:
        raise NativeBuildError("Frozen arithmetic reference has an unsupported layout") from error
    source = re.sub(r"TORCH_CHECK\(.*?\);", "", source, flags=re.S)
    replacements = {
        "check_cdf_16(cdf);": "",
        "check_sym_1d(sym);": "",
        "const int64_t N = cdf.size(0);": "",
        "void encode_chunk(const torch::Tensor& cdf, const torch::Tensor& sym)":
            "void encode_chunk(const uint16_t* cdf, const int16_t* sym, int64_t N)",
        "torch::Tensor decode_chunk(const torch::Tensor& cdf)":
            "std::vector<int16_t> decode_chunk(const uint16_t* cdf, int64_t N)",
        "reinterpret_cast<const cdf_t*>(cdf.data_ptr<int16_t>())": "cdf",
        "sym.accessor<int16_t, 1>()": "sym",
        "auto out = torch::empty({N}, torch::TensorOptions().dtype(torch::kInt16).device(torch::kCPU));":
            "std::vector<int16_t> out(static_cast<size_t>(N));",
        "out.accessor<int16_t, 1>()": "out.data()",
        "py::bytes": "std::string",
    }
    for original, replacement in replacements.items():
        source = source.replace(original, replacement)
    if "torch::" in source or "py::" in source:
        raise NativeBuildError("Frozen arithmetic reference needs a new adapter")
    return (
        "// Adapted from the frozen ELIC arithmetic reference; see LICENSE.\n"
        "#include <cassert>\n#include <cstdint>\n#include <string>\n#include <vector>\n"
        "namespace reference { using cdf_t = uint16_t;\n"
        "static constexpr int kLp=17, kMaxSym=15, kPrecision=16;\n"
        "static constexpr uint32_t kFull=65536U;\n" + source + "\n}\n"
    )


def _compiler_command(compiler: str | None) -> list[str]:
    selected = compiler or os.environ.get("CXX")
    if selected:
        # An existing path can contain spaces without shell quoting.  Otherwise
        # permit the conventional CXX='ccache g++' form without invoking a shell.
        if Path(selected).is_file():
            command = [selected]
        else:
            try:
                command = shlex.split(selected, posix=os.name != "nt")
            except ValueError as error:
                raise NativeUnavailable(f"Invalid compiler command: {error}") from error
            command = [part[1:-1] if len(part) > 1 and part[0] == part[-1] == '"'
                       else part for part in command]
        if not command or not shutil.which(command[0]):
            raise NativeUnavailable(f"C++ compiler not found: {selected}")
        command[0] = str(Path(shutil.which(command[0])).resolve())
        return command
    for name in ("g++", "clang++", "cl"):
        executable = shutil.which(name)
        if executable:
            return [str(Path(executable).resolve())]
    raise NativeUnavailable("C++17 compiler required: install g++, clang++, or MSVC cl; set CXX or --compiler")


def _run(command: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(command, cwd=cwd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout,
                              shell=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise NativeBuildError(f"Cannot run compiler {command[0]}: {error}") from error


@contextmanager
def compiled_library(repo_root: Path, build_dir: Path,
                     compiler: str | None = None) -> Iterator[tuple[ctypes.CDLL, dict]]:
    """Compile/load both coders, yield (CDLL, metadata), then unload and clean up."""
    repo_root, build_dir = Path(repo_root).resolve(), Path(build_dir).resolve()
    core = repo_root / "csrc/arithmetic_core.h"
    reference = repo_root / "benchmarks/reference/csrc/arithmetic_coding.cpp"
    bridge = Path(__file__).with_name("native_bridge.cpp")
    for path in (core, reference, bridge):
        if not path.is_file():
            raise NativeBuildError(f"Required arithmetic source is missing: {path}")
    command_prefix = _compiler_command(compiler)
    msvc = any(Path(item).name.lower() in {"cl", "cl.exe"} for item in command_prefix)
    build_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="elic-native-", dir=build_dir) as temporary:
        build = Path(temporary)
        adapted = build / "reference_core.h"
        adapted.write_text(extract_reference(reference), encoding="utf-8")
        # Copy the measured candidate into the isolated build for a consistent
        # snapshot even if a developer edits the original while testing.
        candidate = build / "arithmetic_core.h"
        candidate.write_bytes(core.read_bytes())
        source = build / "native_bridge.cpp"
        source.write_bytes(bridge.read_bytes())
        suffix = ".dll" if os.name == "nt" else (".dylib" if sys.platform == "darwin" else ".so")
        binary = build / ("elic_arithmetic" + suffix)
        if msvc:
            command = command_prefix + ["/nologo", "/O2", "/EHsc", "/std:c++17", "/LD",
                                        "/DNOMINMAX", f"/I{build}", str(source),
                                        f"/Fo{build / 'native_bridge.obj'}", "/link",
                                        f"/OUT:{binary}", f"/IMPLIB:{build / 'native_bridge.lib'}"]
        else:
            shared = "-dynamiclib" if sys.platform == "darwin" else "-shared"
            command = command_prefix + ["-O3", "-std=c++17", "-DNDEBUG", shared]
            if os.name != "nt":
                command += ["-fPIC"]
            command += [f"-I{build}", str(source), "-o", str(binary)]
        version_result = _run(command_prefix + (["/?"] if msvc else ["--version"]), build, 20)
        started = time.perf_counter()
        completed = _run(command, build, 180)
        elapsed = time.perf_counter() - started
        if completed.returncode or not binary.is_file():
            detail = (completed.stdout + "\n" + completed.stderr).strip()
            raise NativeBuildError(f"C++ arithmetic build failed (exit {completed.returncode}):\n{detail}")
        hashes = {
            "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
            "reference_adapter_sha256": hashlib.sha256(adapted.read_bytes()).hexdigest(),
            "core_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
            "bridge_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        }
        binding = repo_root / "csrc/arithmetic_coding.cpp"
        if binding.is_file():
            hashes["binding_sha256"] = hashlib.sha256(binding.read_bytes()).hexdigest()
        info = {"compiler": command_prefix[0], "compiler_command": command_prefix,
                "compiler_version": (version_result.stdout + "\n" + version_result.stderr).strip(),
                "command": command, "compile_seconds": elapsed, **hashes,
                "build_artifacts_retained": False,
                "scope": "CPU integer arithmetic cores with NumPy/ctypes adapters; excludes Torch, CUDA, neural model and file I/O"}
        directory_handles = []
        library = None
        try:
            if os.name == "nt" and hasattr(os, "add_dll_directory"):
                # MinGW DLL dependencies are normally beside the compiler;
                # user-provided PATH directories may contain further runtimes.
                directories = [Path(command_prefix[0]).parent]
                directories += [Path(item) for item in os.environ.get("PATH", "").split(os.pathsep) if item]
                for directory in dict.fromkeys(directories):
                    if directory.is_dir():
                        try:
                            directory_handles.append(os.add_dll_directory(str(directory)))
                        except OSError:
                            pass
            try:
                library = ctypes.CDLL(str(binary))
            except OSError as error:
                raise NativeUnavailable(f"Arithmetic library built but a C++ runtime could not be loaded: {error}") from error
            yield library, info
        finally:
            if library is not None and library._handle:
                # All native instances must already have been closed by the
                # public build_coders context before this context exits.
                import _ctypes
                if os.name == "nt":
                    _ctypes.FreeLibrary(library._handle)
                else:
                    _ctypes.dlclose(library._handle)
                library._handle = 0
            for handle in reversed(directory_handles):
                handle.close()
