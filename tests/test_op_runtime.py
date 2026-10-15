import argparse
import importlib.util
from pathlib import Path

import pytest
import torch

from src import op
from src.runtime import EntropyEncoder, parse_bool


def reference():
    path = Path(__file__).resolve().parents[1] / 'benchmarks/reference/src/op.py'
    spec = importlib.util.spec_from_file_location('reference_op', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize('normalized', [False, True])
@pytest.mark.parametrize('length', [2, 17, 257])
def test_cdf_exact(dtype, normalized, length):
    rng = torch.Generator().manual_seed(11)
    cdf = torch.rand((1024, length), generator=rng).sort(dim=-1).values.to(dtype)
    cdf[:, 0] = 0
    cdf[:, -1] = 1
    before = cdf.clone()
    expected = reference()._convert_to_int_and_normalize(cdf, normalized)
    assert torch.equal(op._convert_to_int_and_normalize(cdf, normalized), expected)
    assert torch.equal(cdf, before)


def test_cdf_halfway_and_zero_probability():
    cdf = torch.cat((torch.zeros(1, 8), torch.ones(1, 9)), dim=1)
    result = op._convert_to_int_and_normalize(cdf, True)
    assert torch.equal(result, reference()._convert_to_int_and_normalize(cdf, True))
    unsigned = result.to(torch.int32) & 65535
    unsigned[:, -1] = 65536
    assert torch.all(unsigned[:, 1:] > unsigned[:, :-1])
    near = (torch.arange(65520, dtype=torch.float32) + .5) / 65520
    assert torch.equal(op._convert_to_int_and_normalize(near[:, None].expand(-1, 17), True),
                       reference()._convert_to_int_and_normalize(near[:, None].expand(-1, 17), True))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_cdf_exact():
    for dtype in (torch.float32, torch.float64):
        cdf = torch.rand((5000, 17), device='cuda', dtype=dtype).sort(dim=-1).values
        assert torch.equal(op._convert_to_int_and_normalize(cdf, True),
                           reference()._convert_to_int_and_normalize(cdf, True))


def test_bitstream_pack_compatible():
    ref = reference()
    for data in (b'', b'abc\x00\xff', bytes(range(256))):
        encoded = op.pack_bitstream_and_enc_idx_list(data, [5, 5, 0, 3])
        assert encoded == ref.pack_bitstream_and_enc_idx_list(data, [5, 5, 0, 3])
        recovered, indices = op.unpack_bitstream_and_enc_idx_list(encoded)
        assert recovered == data
        assert indices.tolist() == [5, 5, 0, 3]


class RecordingCoder:
    def __init__(self, fail=False):
        self.chunks = []
        self.fail = fail

    def encode_chunk(self, cdf, symbols):
        if self.fail:
            raise ValueError('worker failure')
        self.chunks.append((cdf.clone(), symbols.clone()))

    def finish(self):
        output = b''.join(s.numpy().tobytes() for _, s in self.chunks)
        self.chunks.clear()
        return output


@pytest.mark.parametrize('asynchronous', [False, True])
def test_entropy_order_reset(asynchronous):
    with EntropyEncoder(RecordingCoder(), asynchronous) as encoder:
        for _ in range(3):
            for chunk in range(8):
                encoder.encode_chunk(torch.zeros(2, 17, dtype=torch.int16), torch.tensor([chunk, 15-chunk]))
            output = encoder.finish()
            assert output == torch.tensor([[x, 15-x] for x in range(8)], dtype=torch.int16).numpy().tobytes()


def test_entropy_worker_error_propagates():
    with EntropyEncoder(RecordingCoder(fail=True), True) as encoder:
        encoder.encode_chunk(torch.zeros(1, 17, dtype=torch.int16), torch.tensor([1]))
        with pytest.raises(ValueError, match='worker failure'):
            encoder.finish()


def test_parse_bool():
    for value in ('true', 'YES', '1', True):
        assert parse_bool(value) is True
    for value in ('false', 'NO', '0', False):
        assert parse_bool(value) is False
    with pytest.raises(argparse.ArgumentTypeError):
        parse_bool('maybe')
