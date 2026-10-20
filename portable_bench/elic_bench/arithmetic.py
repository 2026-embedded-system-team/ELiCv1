"""NumPy interface matching the production 16-symbol arithmetic extension.

Inputs are native-endian, C-contiguous NumPy int16 arrays; no implicit cast,
copy, Torch import, or device transfer is performed.  Reinterpret uint16 CDFs
with ``cdf_u16.view(np.int16)``.  Callers own GPU-to-CPU/Torch-to-NumPy conversion.
Build contexts own every native handle: instances cannot outlive the context.
"""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
from pathlib import Path
from types import SimpleNamespace
import weakref

import numpy as np

from .native import NativeBuildError, NativeUnavailable, compiled_library


def _configure(library):
    pointer, i32, i64 = ctypes.c_void_p, ctypes.c_int, ctypes.c_int64
    signatures = {
        "elic_encoder_create": ([i32], pointer),
        "elic_encoder_destroy": ([pointer], None),
        "elic_encoder_reset": ([pointer, i32], None),
        "elic_encoder_chunk": ([pointer, pointer, pointer, i64], i32),
        "elic_encoder_finish_options": ([pointer, i32, i32], i64),
        "elic_encoder_bytes": ([pointer], pointer),
        "elic_encoder_error": ([pointer], ctypes.c_char_p),
        "elic_encoder_buffer": ([pointer], i64),
        "elic_encoder_bitcount": ([pointer], i32),
        "elic_decoder_create": ([i32], pointer),
        "elic_decoder_destroy": ([pointer], None),
        "elic_decoder_reset": ([pointer, ctypes.c_char_p, i64], i32),
        "elic_decoder_chunk": ([pointer, pointer, pointer, i64], i32),
        "elic_decoder_error": ([pointer], ctypes.c_char_p),
        "elic_decoder_consumed": ([pointer], i64),
        "elic_decoder_remaining": ([pointer], i64),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(library, name)
        function.argtypes, function.restype = arguments, result


def _int16_array(value, name, dimensions):
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(np.int16):
        raise TypeError(f"{name} must be a native-endian NumPy int16 array")
    if value.ndim != dimensions:
        raise ValueError(f"{name} must have {dimensions} dimensions")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return value


def validate_cdf(cdf):
    """Validate CDF representation and strictly increasing unsigned boundaries.

    Entry 16 may be 0 (wrapped 65536) or greater than entry 15.  Both cores
    interpret the final upper bound as 65536 regardless of the stored value.
    Returns the same array, without copying.
    """
    cdf = _int16_array(cdf, "CDF", 2)
    if cdf.shape[1] != 17:
        raise ValueError("CDF shape must be (N,17)")
    unsigned = cdf.view(np.uint16)
    if (np.any(unsigned[:, 0] != 0) or
            np.any(unsigned[:, 1:16] <= unsigned[:, :15]) or
            np.any((unsigned[:, 16] != 0) & (unsigned[:, 16] <= unsigned[:, 15]))):
        raise ValueError("CDF must start at 0 and have strictly increasing uint16 boundaries; terminal 0 means 65536")
    return cdf


def _cdf_array(value, validate):
    if validate:
        return validate_cdf(value)
    value = _int16_array(value, "CDF", 2)
    if value.shape[1] != 17:
        raise ValueError("CDF shape must be (N,17)")
    return value


def _module(library, original, state):
    instances = weakref.WeakSet()

    class _Handle:
        handle = None
        _destroy_name = ""

        def _active(self):
            if not state.alive or not self.handle:
                raise RuntimeError("Arithmetic handle is closed; use it inside build_coders()")

        def close(self):
            if self.handle:
                if state.alive:
                    getattr(library, self._destroy_name)(self.handle)
                self.handle = None

        def __del__(self):
            self.close()

        def __enter__(self):
            self._active()
            return self

        def __exit__(self, *args):
            self.close()

    class ArithmeticEncoder16(_Handle):
        """Stateful encoder; validate=False skips only CDF value checks."""
        _destroy_name = "elic_encoder_destroy"

        def __init__(self, validate=True):
            if not state.alive:
                raise RuntimeError("build_coders() context has ended")
            self.validate = bool(validate)
            self.handle = library.elic_encoder_create(original)
            if not self.handle:
                raise MemoryError("Cannot create arithmetic encoder")
            instances.add(self)

        def encode_chunk(self, cdf, symbols):
            self._active()
            cdf = _cdf_array(cdf, self.validate)
            symbols = _int16_array(symbols, "Symbols", 1)
            if len(symbols) != len(cdf):
                raise ValueError("Symbol length must match CDF rows")
            # The Torch check was removed from the adapted original; validate
            # both paths identically before either native core sees symbols.
            if np.any(symbols < 0) or np.any(symbols > 15):
                raise ValueError("Symbols must be in range 0..15")
            if library.elic_encoder_chunk(self.handle, cdf.ctypes.data, symbols.ctypes.data, len(symbols)):
                raise RuntimeError(library.elic_encoder_error(self.handle).decode("utf-8", "replace"))

        def reset(self, reset_output=True):
            self._active()
            library.elic_encoder_reset(self.handle, bool(reset_output))

        def finish(self, reset_after=True, clear_output_after=True):
            self._active()
            size = library.elic_encoder_finish_options(self.handle, bool(reset_after), bool(clear_output_after))
            if size < 0:
                raise RuntimeError(library.elic_encoder_error(self.handle).decode("utf-8", "replace"))
            return ctypes.string_at(library.elic_encoder_bytes(self.handle), size)

        def get_buffer_bytes(self):
            self._active()
            size = library.elic_encoder_buffer(self.handle)
            return ctypes.string_at(library.elic_encoder_bytes(self.handle), size)

        def buffered_bitcount(self):
            self._active()
            return library.elic_encoder_bitcount(self.handle)

    class ArithmeticDecoder16(_Handle):
        """Stateful decoder; reset(bytes) starts an independent stream."""
        _destroy_name = "elic_decoder_destroy"

        def __init__(self, validate=True):
            if not state.alive:
                raise RuntimeError("build_coders() context has ended")
            self.validate = bool(validate)
            self.handle = library.elic_decoder_create(original)
            if not self.handle:
                raise MemoryError("Cannot create arithmetic decoder")
            instances.add(self)

        def reset(self, stream):
            self._active()
            if not isinstance(stream, bytes):
                raise TypeError("Bitstream must be bytes")
            if library.elic_decoder_reset(self.handle, stream, len(stream)):
                raise RuntimeError(library.elic_decoder_error(self.handle).decode("utf-8", "replace"))

        def decode_chunk(self, cdf):
            self._active()
            cdf = _cdf_array(cdf, self.validate)
            symbols = np.empty(len(cdf), dtype=np.int16)
            if library.elic_decoder_chunk(self.handle, cdf.ctypes.data, symbols.ctypes.data, len(symbols)):
                raise RuntimeError(library.elic_decoder_error(self.handle).decode("utf-8", "replace"))
            return symbols

        def bytes_consumed(self):
            self._active()
            return library.elic_decoder_consumed(self.handle)

        def bytes_remaining(self):
            self._active()
            return library.elic_decoder_remaining(self.handle)

    def encode_cdf_16(cdf, sym):
        with ArithmeticEncoder16() as encoder:
            encoder.encode_chunk(cdf, sym)
            return encoder.finish(False, False)

    def decode_cdf_16(cdf, bitstream):
        with ArithmeticDecoder16() as decoder:
            decoder.reset(bitstream)
            return decoder.decode_chunk(cdf)

    return SimpleNamespace(ArithmeticEncoder16=ArithmeticEncoder16,
                           ArithmeticDecoder16=ArithmeticDecoder16,
                           encode_cdf_16=encode_cdf_16, decode_cdf_16=decode_cdf_16,
                           _instances=instances)


@contextmanager
def build_coders(repo_root: Path, build_dir: Path, compiler: str | None = None):
    """Yield (frozen reference, production candidate, build metadata).

    CXX selects a compiler when compiler is None.  Discovery otherwise tries
    g++, clang++, then cl.  Temporary sources/binaries are removed on exit,
    after closing all coder instances and unloading the library on every OS.
    Compilation/runtime-unavailable failures are never converted to passes.
    """
    with compiled_library(repo_root, build_dir, compiler) as (library, info):
        _configure(library)
        state = SimpleNamespace(alive=True)
        modules = (_module(library, True, state), _module(library, False, state))
        try:
            yield modules[0], modules[1], info
        finally:
            for module in modules:
                for instance in list(module._instances):
                    instance.close()
            state.alive = False


def _benchmark_fixtures(points, seed):
    """Make deterministic valid CDFs without depending on Torch or data files."""
    rng = np.random.default_rng(seed)
    n = max(1, min(int(points), 65536))

    def cdf_from_weights(weights):
        sums = np.cumsum(weights, axis=1, dtype=np.float64)
        cumulative = np.column_stack((np.zeros(n), sums / sums[:, -1, None]))
        # Reserve one integer count per symbol, as the production normalizer
        # does, so even almost-zero probabilities have valid intervals.
        unsigned = (np.floor(cumulative * 65520).astype(np.uint32) +
                    np.arange(17, dtype=np.uint32)).astype(np.uint16)
        result = np.ascontiguousarray(unsigned.view(np.int16))
        validate_cdf(result)
        return result

    weights = np.ones((n, 16), dtype=np.float64)
    symbols = rng.integers(0, 16, n, dtype=np.int16)
    yield "uniform", cdf_from_weights(weights), symbols
    weights[:, 0] = 100000
    concentrated = np.zeros(n, dtype=np.int16)
    concentrated[::101] = np.arange(len(concentrated[::101]), dtype=np.int16) % 16
    yield "concentrated", cdf_from_weights(weights), concentrated
    weights = rng.integers(1, 1000, (n, 16)).astype(np.float64)
    yield "random_weighted", cdf_from_weights(weights), rng.integers(0, 16, n, dtype=np.int16)
    weights.fill(1)
    weights[:, 15] = 50000
    rare = np.full(n, 15, dtype=np.int16)
    rare[::127] = 0
    rare[31::131] = 14
    yield "rare_symbols", cdf_from_weights(weights), rare


def run_arithmetic(ctx, reference, candidate):
    """Yield exactness and paired CPU arithmetic timing case records.

    ``ctx`` follows the portable Context contract.  Fixtures and CDF value
    validation happen outside timing.  Calls retain NumPy shape/type checks,
    encoder symbol bounds checks, ctypes crossings, native object construction,
    and returned bytes/array allocation.  No Torch extension or CUDA kernel is
    exercised by these core cases, even when other suites select a GPU.
    """
    scope = ("CPU integer arithmetic coder plus NumPy/ctypes adapter; includes shape/dtype checks, "
             "symbol range checks, native object and output allocation; excludes fixture creation, "
             "CDF value validation, compilation, file I/O, Torch binding and CUDA")
    fixtures = list(_benchmark_fixtures(ctx.config["points"], ctx.config["seed"]))

    def encode(module, cdf, symbols):
        with module.ArithmeticEncoder16(validate=False) as encoder:
            encoder.encode_chunk(cdf, symbols)
            return encoder.finish()

    def decode(module, cdf, stream):
        with module.ArithmeticDecoder16(validate=False) as decoder:
            decoder.reset(stream)
            return decoder.decode_chunk(cdf)

    for label, cdf, symbols in fixtures:
        details = {"uses_cuda": False, "symbols": len(symbols), "requested_symbols": ctx.config["points"],
                   "fixture_max_symbols": 65536, "distribution": label, "scope": scope,
                   "timing_scope": scope, "cdf_value_validation": "once before timing"}

        def check_streams(old, new):
            assert isinstance(old, bytes) and isinstance(new, bytes)
            assert old == new, "Arithmetic bitstreams differ"
            for module in (reference, candidate):
                for stream in (old, new):
                    np.testing.assert_array_equal(decode(module, cdf, stream), symbols)
            return {"bitstream_exact": True, "symbol_accuracy": 1.0,
                    "exact_symbols": len(symbols), "expected_symbols": len(symbols),
                    "coder_combinations_passed": 4, "stream_bytes": len(new)}

        yield ctx.record_case(f"arithmetic.encode.{label}", "arithmetic",
                              lambda: encode(reference, cdf, symbols),
                              lambda: encode(candidate, cdf, symbols),
                              check=check_streams, details=details)
        old_stream, new_stream = encode(reference, cdf, symbols), encode(candidate, cdf, symbols)

        def check_symbols(old, new):
            np.testing.assert_array_equal(old, symbols)
            np.testing.assert_array_equal(new, symbols)
            assert old_stream == new_stream, "Arithmetic bitstreams differ"
            np.testing.assert_array_equal(decode(reference, cdf, new_stream), symbols)
            np.testing.assert_array_equal(decode(candidate, cdf, old_stream), symbols)
            return {"symbol_accuracy": 1.0, "exact_symbols": len(symbols),
                    "expected_symbols": len(symbols), "coder_combinations_passed": 4,
                    "bitstream_exact": True}

        yield ctx.record_case(f"arithmetic.decode.{label}", "arithmetic",
                              lambda: decode(reference, cdf, old_stream),
                              lambda: decode(candidate, cdf, new_stream),
                              check=check_symbols, details=dict(details, stream_bytes=len(new_stream)))

    cdf, symbols = fixtures[2][1][:67], fixtures[2][2][:67]

    def streaming_contracts():
        streams = []
        for module in (reference, candidate):
            expected = module.encode_cdf_16(cdf, symbols)
            with module.ArithmeticEncoder16() as encoder:
                for start, end in ((0, 0), (0, 1), (1, 13), (13, len(symbols))):
                    encoder.encode_chunk(cdf[start:end], symbols[start:end])
                stream = encoder.finish()
                assert stream == expected
                assert encoder.get_buffer_bytes() == b"" and encoder.buffered_bitcount() == 0
                encoder.encode_chunk(cdf, symbols)
                assert encoder.finish() == stream
                encoder.reset()
                assert encoder.get_buffer_bytes() == b""
            with module.ArithmeticDecoder16() as decoder:
                decoder.reset(stream)
                pieces = (decoder.decode_chunk(cdf[:7]), decoder.decode_chunk(cdf[7:]))
                np.testing.assert_array_equal(np.concatenate(pieces), symbols)
                decoder.reset(stream)
                np.testing.assert_array_equal(decoder.decode_chunk(cdf), symbols)
            np.testing.assert_array_equal(module.decode_cdf_16(cdf, expected), symbols)
            empty = module.encode_cdf_16(cdf[:0], symbols[:0])
            assert module.decode_cdf_16(cdf[:0], empty).shape == (0,)
            streams.append((stream, empty))
        assert streams[0] == streams[1]
        return True

    def options_contracts():
        for reset_after in (False, True):
            for clear_output_after in (False, True):
                results = []
                for module in (reference, candidate):
                    with module.ArithmeticEncoder16() as encoder:
                        encoder.encode_chunk(cdf, symbols)
                        first = encoder.finish(reset_after, clear_output_after)
                        buffer = encoder.get_buffer_bytes()
                        encoder.reset(reset_output=False)
                        encoder.encode_chunk(cdf, symbols)
                        second = encoder.finish()
                        results.append((first, buffer, second))
                assert results[0] == results[1], "Nondefault finish/reset behavior differs"
        return True

    def invalid_input_contracts():
        rejected = 0
        for module in (reference, candidate):
            with module.ArithmeticEncoder16() as encoder:
                cases = [(cdf, symbols.astype(np.int32)), (cdf, symbols[:-1]),
                         (cdf[:, :16].copy(), symbols), (cdf[:, ::-1], symbols)]
                for value in (-1, 16):
                    bad = symbols.copy()
                    bad[0] = value
                    cases.append((cdf, bad))
                invalid = cdf.copy()
                invalid[0, 2] = invalid[0, 1]
                cases.append((invalid, symbols))
                for input_cdf, input_symbols in cases:
                    try:
                        encoder.encode_chunk(input_cdf, input_symbols)
                    except (TypeError, ValueError):
                        rejected += 1
                    else:
                        raise AssertionError("Invalid arithmetic input was accepted")
                encoder.encode_chunk(cdf, symbols)
                assert encoder.finish() == module.encode_cdf_16(cdf, symbols)
        assert rejected == 14
        return True

    for label, check in (("streaming_reset_empty", streaming_contracts),
                         ("finish_options", options_contracts),
                         ("invalid_inputs", invalid_input_contracts)):
        yield ctx.record_case(f"arithmetic.contract.{label}", "arithmetic", lambda: True, check,
                              details={"uses_cuda": False, "timing_scope": "correctness_only",
                                       "scope": "NumPy adapter contract and frozen/production core compatibility"})

    if ctx.config.get("async_entropy"):
        yield _async_entropy_case(ctx, candidate, cdf, symbols)


def _async_entropy_case(ctx, candidate, cdf, symbols):
    """Exercise the real runtime worker with an explicit Torch/NumPy adapter."""
    import importlib.util
    try:
        import torch
    except ImportError as error:
        return ctx.add_result("arithmetic.contract.async_entropy", "arithmetic", "skipped",
                              reason=f"Torch is required for the runtime entropy interface: {error}")

    def check():
        runtime_path = ctx.repo_root / "src/runtime.py"
        spec = importlib.util.spec_from_file_location("_elic_bench_entropy_runtime", runtime_path)
        runtime = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runtime)
        tcdf = torch.from_numpy(cdf).to(ctx.device)
        tsymbols = torch.from_numpy(symbols).to(ctx.device)
        expected = candidate.encode_cdf_16(cdf, symbols)

        class TorchToNumpyCoder:
            def __init__(self, coder):
                self.coder = coder

            def encode_chunk(self, chunk_cdf, chunk_symbols):
                # EntropyEncoder already transfers to CPU/contiguous.  The
                # explicit conversion aligns this optional test's boundary.
                self.coder.encode_chunk(chunk_cdf.detach().numpy(), chunk_symbols.detach().numpy())

            def finish(self):
                return self.coder.finish()

        for asynchronous in (False, True):
            with candidate.ArithmeticEncoder16() as coder:
                with runtime.EntropyEncoder(TorchToNumpyCoder(coder), asynchronous=asynchronous) as worker:
                    for start, end in ((0, 1), (1, 7), (7, 13), (13, len(symbols))):
                        worker.encode_chunk(tcdf[start:end], tsymbols[start:end])
                    assert worker.finish() == expected, "Runtime entropy worker changed chunk ordering"
        return True

    return ctx.record_case("arithmetic.contract.async_entropy", "arithmetic", lambda: True, check,
                           details={"uses_cuda": ctx.device.type == "cuda", "timing_scope": "correctness_only",
                                    "scope": "real src.runtime.EntropyEncoder sync/async ordering; explicit Torch-to-NumPy adapter; no Torch extension build or GIL-overlap speed claim"})
