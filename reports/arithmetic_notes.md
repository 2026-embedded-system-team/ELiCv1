# Arithmetic-coder optimization and validation

The production Torch extension delegates its integer operations to
`csrc/arithmetic_core.h`. Its streaming API, 32-bit interval, 16-bit CDF precision,
symbol order, padding and finish/reset behavior are preserved. Optimizations:

- Keep interval state in local variables while processing a chunk.
- Batch consecutive E1/E2 and E3 normalization steps using leading-zero counts;
  buffer bits by byte instead of calling bit I/O once per emitted/read bit.
- Search balanced CDFs with four integer multiply/comparison steps, using
  `cdf[s] * span <= ((value - low + 1) << 16) - 1` instead of division. Very
  skewed CDFs retain the original division/binary search, which is faster there.
- Release Python's GIL around the pure Tensor encode/decode chunk methods.
  Python byte creation and stream reset still hold the GIL.

## Measured CPU results

These are isolated integer-core measurements with g++ `-O3 -std=c++17`, not
GPU neural-network or end-to-end codec measurements. Each row uses nine
alternating old/new pairs after warm-up; ratios are baseline median / optimized
median. Both versions include equivalent output-vector allocation and stream
copying. `arithmetic_benchmark.json` contains all 36 raw timing observations per
row, source hashes, corpus details, and medians.

| Input | Encode speedup | Decode speedup |
| --- | ---: | ---: |
| Uniform synthetic CDF | 3.463x | 1.423x |
| Extremely concentrated synthetic CDF | 0.945x | 1.141x |
| Random weighted synthetic CDF | 1.680x | 1.805x |
| Random low-probability synthetic symbols | 2.002x | 2.027x |
| Ford sample 0001, empirical occupancy CDF | 1.371x | 1.355x |
| Ford sample 0002, empirical occupancy CDF | 2.006x | 1.295x |
| SemanticKITTI sample 0001, empirical occupancy CDF | 1.584x | 1.421x |
| SemanticKITTI sample 0002, empirical occupancy CDF | 1.715x | 1.413x |

The concentrated encode case takes approximately 5.9% longer. This remaining
regression is included rather than claiming that every distribution improves.
Real sample rows use actual quantized hierarchy occupancy symbols and their
per-layer empirical histograms; **they do not use the model's predicted CDFs**.
The four clouds contain 80,278–126,012 input points, 12 layers each, and
473,004–496,866 two-stage symbols each, at position quantization 16.

## Correctness and reproduction

The frozen original `benchmarks/reference/csrc/arithmetic_coding.cpp` is not
modified. The runner extracts its arithmetic code automatically, replaces
Torch/Python container adapters, and removes Torch validation macros. The
optimized production core retains its symbol-range check, so removing these
checks from the baseline does not inflate the reported speedup. Tests compare
the baseline arithmetic with the exact production header, avoiding a separate
handwritten optimized implementation.

447 synthetic/boundary cases and four empirical cloud-symbol streams pass
byte-for-byte encoding comparisons and decoding in both directions. Coverage
includes every symbol, unit-frequency CDF intervals, random CDFs, empty/single
streams, different chunk partitions, intermediate byte buffers/bit counts,
encoder/decoder reset, consumed-byte counts, and thousands of pending E3 bits.
Four pytest checks also cover the ctypes interface, the real C++ core inside
the ordered asynchronous `EntropyEncoder` worker, nondefault finish/reset flags,
and empty-stream decoder behavior. All four passed in the final full suite.

```powershell
python -m pytest tests/test_arithmetic.py -q
python benchmarks/benchmark_arithmetic.py --benchmark --geometry-histograms --output reports/arithmetic_benchmark.json
python benchmarks/check_torch_binding.py --output reports/arithmetic_torch_binding.json
```

The first two commands require NumPy plus g++/clang++; pytest runtime integration
also requires PyTorch. The last command builds the actual Torch wrapper without
Ninja. On this machine it fails because the available MSVC environment cannot
find the Windows SDK/UCRT `assert.h` header (`C1083`). No packages were installed.
The failure is recorded in `arithmetic_torch_binding.json` and
`arithmetic_torch_build.log`. Actual Torch extension compilation/API execution and
CUDA neural codec round-trips therefore remain unverified in this environment.
