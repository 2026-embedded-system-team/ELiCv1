"""Actual LiDAR hierarchy roundtrip using deterministic, non-neural CDFs.

This checks composition of geometry and the production integer coder. It does
not validate the trained TorchSparse neural probability model or GPU execution.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmarks.arithmetic_bridge import build_coders
from src import io, morton, op

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def coders():
    with build_coders() as modules:
        yield modules


def cdf_for(coords, stage):
    # Probability tables depend only on the already decoded coordinates and
    # stage. Nonuniform rows exercise all 16-bit arithmetic interval updates.
    values = torch.arange(1, 17, dtype=torch.float32)[None, :]
    shift = ((coords[:, 1:].sum(1) + stage) % 16).to(torch.int64)
    weights = values.expand(len(coords), -1).gather(1, (torch.arange(16)[None, :] + shift[:, None]) % 16)
    probs = weights / weights.sum(dim=-1, keepdim=True)
    cumulative = torch.cat((probs[:, :1]*0, probs.cumsum(dim=-1)), dim=-1)
    return op._convert_to_int_and_normalize(cumulative, True).numpy()


def quantize(path, posQ):
    points = io.read_points(path)
    xyz = torch.tensor(points if 'Ford' in str(path) else points / .001 + 131072)
    if 'Ford' in str(path) and xyz.min() < 0:
        xyz += 131072
    xyz = torch.round(xyz / posQ).int()
    return torch.cat((xyz[:, :1]*0, xyz), dim=1)


@pytest.mark.parametrize('path', sorted((ROOT/'data').glob('*Example/*.ply')), ids=lambda p: str(p.relative_to(ROOT/'data')))
@pytest.mark.parametrize('posQ', [4, 8, 16, 32, 64])
def test_real_geometry_entropy_cross_decode(coders, path, posQ):
    old, new = coders
    coords = quantize(path, posQ)
    hierarchy = morton.build_hierarchy(coords)
    streams = []
    for module in (old, new):
        encoder = module.ArithmeticEncoder16()
        for parent, occ in hierarchy:
            for stage in (0, 1):
                symbols = (occ & 15) if stage == 0 else (occ >> 4)
                encoder.encode_chunk(cdf_for(parent, stage), symbols.to(torch.int16).numpy())
        streams.append(encoder.finish())
    assert streams[0] == streams[1]
    expected, _ = morton.morton3_sort(coords)
    # Exercise all four original/optimized encoder/decoder combinations.
    for module in (old, new):
        decoder = module.ArithmeticDecoder16()
        for stream in streams:
            decoder.reset(stream)
            parents = hierarchy[0][0]
            for original_parent, original_occ in hierarchy:
                assert torch.equal(parents, original_parent)
                s0 = decoder.decode_chunk(cdf_for(parents, 0))
                s1 = decoder.decode_chunk(cdf_for(parents, 1))
                occ = torch.from_numpy(s0.astype(np.int32) + s1.astype(np.int32)*16)
                assert torch.equal(occ, original_occ)
                parents = morton.upscale_coordinate(parents, occ)
            assert torch.equal(parents, expected)


def test_frozen_baseline_integrity():
    import hashlib
    import json
    folder = ROOT/'benchmarks/reference'
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    for path, digest in manifest['files'].items():
        assert hashlib.sha256((folder/path).read_bytes()).hexdigest() == digest, path


def test_neural_primitives_unchanged():
    # No retraining, approximate arithmetic, or changes to convolution weights.
    assert (ROOT/'src/nn.py').read_bytes() == (ROOT/'benchmarks/reference/src/nn.py').read_bytes()
