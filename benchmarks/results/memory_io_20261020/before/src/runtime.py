"""Shared codec runtime; entropy submissions always keep their stream order."""
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch


def load_arithmetic_coder():
    from torch.utils.cpp_extension import load
    root = Path(__file__).resolve().parents[1]
    build = root / '.build' / 'arithmetic'
    build.mkdir(parents=True, exist_ok=True)
    return load(
        name="elic_ac_ext",
        sources=[str(root / "csrc/arithmetic_coding.cpp")],
        build_directory=str(build),
        extra_cflags=["/O2", "/std:c++17"] if os.name == "nt" else ["-O3", "-std=c++17"],
        with_cuda=False,
        verbose=False,
    )


class EntropyEncoder:
    """At most two CPU chunks in flight, with a single owner of coder state.

    CUDA tensors are copied synchronously before submission. The worker can
    encode depth b while the caller launches neural computation for b+1.
    The C++ encode_chunk binding must release the GIL for this overlap.
    """
    def __init__(self, coder, asynchronous=False):
        self.coder = coder
        self.pool = ThreadPoolExecutor(max_workers=1) if asynchronous else None
        self.pending = []

    def encode_chunk(self, cdf, symbols):
        cdf = cdf.to(device="cpu").contiguous()
        symbols = symbols.to(device="cpu", dtype=torch.int16).contiguous()
        if self.pool is None:
            self.coder.encode_chunk(cdf, symbols)
            return
        if len(self.pending) >= 2:
            self.pending.pop(0).result()
        # Own these tensor references until completion; callers do not mutate
        # them. PyTorch's inference mode is thread-local, so set it in worker.
        self.pending.append(self.pool.submit(self._encode, cdf, symbols))

    def _encode(self, cdf, symbols):
        with torch.inference_mode():
            self.coder.encode_chunk(cdf, symbols)

    def finish(self):
        for future in self.pending:
            future.result()
        self.pending.clear()
        return self.coder.finish()

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True)
            self.pool = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "ELiCv1's TorchSparse neural codec requires CUDA. "
            "CPU module tests: python -m pytest tests; "
            "benchmarks: python benchmarks/bench_modules.py --device cpu"
        )
    return torch.device("cuda")


def parse_bool(value):
    import argparse
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes"):
        return True
    if value in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError("expected true or false")
