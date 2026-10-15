"""NumPy/ctypes bridge for original and optimized production integer coders.

Example: ``with build_coders() as coders: ...`` yields (old, new) modules with
ArithmeticEncoder16/ArithmeticDecoder16 APIs matching the Torch extension.
These accept contiguous NumPy int16 CDFs and symbols instead of Torch tensors.
"""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import weakref

import numpy as np

from benchmarks.benchmark_arithmetic import ROOT, extract_reference


def _module(library, original):
    instances = weakref.WeakSet()
    pointer = ctypes.c_void_p
    library.elic_encoder_create.argtypes = [ctypes.c_int]
    library.elic_encoder_create.restype = pointer
    library.elic_encoder_destroy.argtypes = [pointer]
    library.elic_encoder_reset.argtypes = [pointer, ctypes.c_int]
    library.elic_encoder_chunk.argtypes = [pointer, pointer, pointer, ctypes.c_int64]
    library.elic_encoder_chunk.restype = ctypes.c_int
    library.elic_encoder_finish.argtypes = [pointer]
    library.elic_encoder_finish.restype = ctypes.c_int64
    library.elic_encoder_finish_options.argtypes = [pointer, ctypes.c_int, ctypes.c_int]
    library.elic_encoder_finish_options.restype = ctypes.c_int64
    library.elic_encoder_bytes.argtypes = [pointer]
    library.elic_encoder_bytes.restype = pointer
    library.elic_encoder_error.argtypes = [pointer]
    library.elic_encoder_error.restype = ctypes.c_char_p
    library.elic_decoder_create.argtypes = [ctypes.c_int]
    library.elic_decoder_create.restype = pointer
    library.elic_decoder_destroy.argtypes = [pointer]
    library.elic_decoder_reset.argtypes = [pointer, ctypes.c_char_p, ctypes.c_int64]
    library.elic_decoder_reset.restype = ctypes.c_int
    library.elic_decoder_chunk.argtypes = [pointer, pointer, pointer, ctypes.c_int64]
    library.elic_decoder_chunk.restype = ctypes.c_int
    library.elic_decoder_error.argtypes = [pointer]
    library.elic_decoder_error.restype = ctypes.c_char_p

    def cdf_array(value):
        value = np.ascontiguousarray(value, dtype=np.int16)
        if value.ndim != 2 or value.shape[1] != 17:
            raise ValueError("CDF shape must be (N,17)")
        return value

    class ArithmeticEncoder16:
        def __init__(self):
            self.handle = library.elic_encoder_create(original)
            if not self.handle:
                raise MemoryError("Cannot create arithmetic encoder")
            instances.add(self)

        def __del__(self):
            if getattr(self, "handle", None):
                library.elic_encoder_destroy(self.handle)
                self.handle = None

        def encode_chunk(self, cdf, symbols):
            cdf = cdf_array(cdf)
            symbols = np.ascontiguousarray(symbols, dtype=np.int16)
            if symbols.ndim != 1 or len(symbols) != len(cdf):
                raise ValueError("Symbol shape must match CDF rows")
            if library.elic_encoder_chunk(self.handle, cdf.ctypes.data, symbols.ctypes.data, len(symbols)):
                raise RuntimeError(library.elic_encoder_error(self.handle).decode())

        def reset(self, reset_output=True):
            library.elic_encoder_reset(self.handle, reset_output)

        def finish(self, reset_after=True, clear_output_after=True):
            size = library.elic_encoder_finish_options(self.handle, reset_after, clear_output_after)
            if size < 0:
                raise RuntimeError(library.elic_encoder_error(self.handle).decode())
            return ctypes.string_at(library.elic_encoder_bytes(self.handle), size)

    class ArithmeticDecoder16:
        def __init__(self):
            self.handle = library.elic_decoder_create(original)
            if not self.handle:
                raise MemoryError("Cannot create arithmetic decoder")
            instances.add(self)

        def __del__(self):
            if getattr(self, "handle", None):
                library.elic_decoder_destroy(self.handle)
                self.handle = None

        def reset(self, stream):
            if library.elic_decoder_reset(self.handle, stream, len(stream)):
                raise RuntimeError(library.elic_decoder_error(self.handle).decode())

        def decode_chunk(self, cdf):
            cdf = cdf_array(cdf)
            symbols = np.empty(len(cdf), dtype=np.int16)
            if library.elic_decoder_chunk(self.handle, cdf.ctypes.data, symbols.ctypes.data, len(symbols)):
                raise RuntimeError(library.elic_decoder_error(self.handle).decode())
            return symbols

    return SimpleNamespace(ArithmeticEncoder16=ArithmeticEncoder16,
                           ArithmeticDecoder16=ArithmeticDecoder16, _instances=instances)


@contextmanager
def build_coders(compiler=None):
    compiler = compiler or shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        raise RuntimeError("g++ or clang++ required to build the optional standalone bridge")
    build_root = (ROOT / "reports/.arithmetic_tmp").resolve()
    build_root.relative_to(ROOT.resolve())
    build_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="elic-arithmetic-bridge-", dir=build_root) as temp:
        build = Path(temp)
        (build / "reference_core.h").write_text(extract_reference(), encoding="utf-8")
        binary = build / ("arithmetic_bridge.dll" if os.name == "nt" else "arithmetic_bridge.so")
        command = [compiler, "-O3", "-std=c++17", "-shared", f"-I{build}", f"-I{ROOT / 'csrc'}",
                   str(ROOT / "benchmarks/arithmetic_ctypes.cpp"), "-o", str(binary)]
        if os.name != "nt":
            command.insert(1, "-fPIC")
        subprocess.run(command, check=True, capture_output=True, text=True)
        # Python 3.8+ uses explicit DLL search directories on Windows.
        dll_directory = os.add_dll_directory(str(Path(compiler).resolve().parent)) if os.name == "nt" else None
        library = ctypes.CDLL(str(binary))
        modules = (_module(library, True), _module(library, False))
        try:
            yield modules
        finally:
            for module in modules:
                for instance in list(module._instances):
                    instance.__del__()
            # Windows cannot remove a loaded DLL when TemporaryDirectory exits.
            if os.name == "nt":
                import _ctypes
                _ctypes.FreeLibrary(library._handle)
                dll_directory.close()
