"""Arithmetic format compatibility checks without TorchSparse or CUDA."""
import shutil

import numpy as np
import pytest

from benchmarks.benchmark_arithmetic import run
from benchmarks.arithmetic_bridge import build_coders


pytestmark = pytest.mark.skipif(
    not (shutil.which("g++") or shutil.which("clang++")),
    reason="standalone arithmetic checks require g++ or clang++",
)


def test_original_production_integer_core_compatibility():
    result = run()
    assert result["correctness"] == "passed"
    assert result["cases"] == 447
    assert not result["benchmarks"]


def test_ctypes_chunk_api_and_reset():
    cdf = np.tile((np.arange(17, dtype=np.uint32) * 4096).astype(np.int16), (256, 1))
    symbols = np.tile(np.arange(16, dtype=np.int16), 16)
    with build_coders() as (old, new):
        encoders = [old.ArithmeticEncoder16(), new.ArithmeticEncoder16()]
        for encoder in encoders:
            encoder.encode_chunk(cdf[:33], symbols[:33])
            encoder.encode_chunk(cdf[33:33], symbols[33:33])
            encoder.encode_chunk(cdf[33:], symbols[33:])
        streams = [encoder.finish() for encoder in encoders]
        assert streams[0] == streams[1]
        for module, stream in [(new, streams[0]), (old, streams[1])]:
            decoder = module.ArithmeticDecoder16()
            for _ in range(2):
                decoder.reset(stream)
                decoded = np.concatenate([decoder.decode_chunk(cdf[:71]), decoder.decode_chunk(cdf[71:])])
                np.testing.assert_array_equal(decoded, symbols)
        for encoder in encoders:
            encoder.encode_chunk(cdf, symbols)
            assert encoder.finish() == streams[0]
        with pytest.raises(RuntimeError, match="sym out of range"):
            encoders[1].encode_chunk(cdf[:1], np.array([16], dtype=np.int16))


def test_async_entropy_worker_preserves_actual_core_stream_order():
    import torch
    from src.runtime import EntropyEncoder

    cdf = torch.from_numpy(np.tile((np.arange(17, dtype=np.uint32) * 4096).astype(np.int16), (2048, 1)))
    symbols = torch.arange(2048, dtype=torch.int16) % 16
    with build_coders() as (_, optimized):
        expected_encoder = optimized.ArithmeticEncoder16()
        expected_encoder.encode_chunk(cdf.numpy(), symbols.numpy())
        expected = expected_encoder.finish()
        with EntropyEncoder(optimized.ArithmeticEncoder16(), asynchronous=True) as encoder:
            for start in range(0, len(symbols), 113):
                encoder.encode_chunk(cdf[start:start + 113], symbols[start:start + 113])
            assert encoder.finish() == expected


def test_nondefault_finish_reset_and_empty_decoder_semantics():
    cdf = np.tile((np.arange(17, dtype=np.uint32) * 4096).astype(np.int16), (16, 1))
    symbols = np.arange(16, dtype=np.int16)
    with build_coders() as (old, new):
        encoders = [old.ArithmeticEncoder16(), new.ArithmeticEncoder16()]
        for encoder in encoders:
            encoder.encode_chunk(cdf[:3], symbols[:3])
            encoder.reset(False)  # Preserve a partly filled byte.
            encoder.encode_chunk(cdf[3:7], symbols[3:7])
        for reset_after, clear_output in [(False, False), (False, True), (True, False), (True, True)]:
            assert encoders[0].finish(reset_after, clear_output) == encoders[1].finish(reset_after, clear_output)
            for encoder in encoders:
                encoder.encode_chunk(cdf[7:11], symbols[7:11])
        for encoder in encoders:
            encoder.reset(True)
            encoder.encode_chunk(cdf, symbols)
        assert encoders[0].finish() == encoders[1].finish()
        old_decoder, new_decoder = old.ArithmeticDecoder16(), new.ArithmeticDecoder16()
        np.testing.assert_array_equal(old_decoder.decode_chunk(cdf), new_decoder.decode_chunk(cdf))
        old_decoder.reset(b"")
        new_decoder.reset(b"")
        decoded_old, decoded_new = old_decoder.decode_chunk(cdf), new_decoder.decode_chunk(cdf)
        np.testing.assert_array_equal(decoded_old, decoded_new)
        np.testing.assert_array_equal(decoded_new, np.zeros(16, dtype=np.int16))
