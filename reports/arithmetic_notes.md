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

## Historical CPU results (frozen original baseline)

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

## Incremental memory-I/O update (2026-10-20)

`BitReader` now loads four adjacent bytes with a bounded `memcpy` into a 64-bit
bit reservoir, reducing refill operations and branches. Explicit byte order,
the final zero padding, and logical consumed-byte counts are preserved; the
last zero to three bytes use bounded byte loads.

This comparison uses the immediately preceding optimized header saved in
`benchmarks/results/memory_io_20261020/before/csrc/arithmetic_core.h`, not the
frozen original above. It measures the single-threaded CPU integer decoder with
g++ 15.2.0, `-O3 -std=c++17 -DNDEBUG`, three warm-ups and 15 alternating
before/after pairs. Timing includes decoder reset/stream copy and `decode_chunk`;
both decoders use equally preallocated outputs. CDF/input generation and
compilation are excluded. The real samples use position quantization 16 and
per-layer empirical occupancy CDFs, not neural-network predictions.

| Input | Before median (ms) | After median (ms) | Speedup |
| --- | ---: | ---: | ---: |
| Ford sample 0001 | 12.0767 | 10.6722 | 1.132x |
| Ford sample 0002 | 11.2977 | 9.9293 | 1.138x |
| SemanticKITTI sample 0001 | 11.7730 | 10.3471 | 1.138x |
| SemanticKITTI sample 0002 | 13.7863 | 11.8405 | 1.164x |
| Sum of the four sample medians | **48.9337** | **42.7891** | **1.1436x** |

The sum falls by **12.56%**; this aggregate is a ratio of summed per-sample
medians, not an average speedup. Uniform synthetic input is 0.993x
(18.2594 to 18.3967 ms, about 0.75% slower), and concentrated input is 1.005x
(6.6669 to 6.6342 ms), so neither shows a clear gain. Random weighted and
low-probability synthetic inputs reach 1.136x and 1.162x respectively. These
incremental figures must not be multiplied by or merged with the historical
timings. They establish CPU integer-decoder performance only; CUDA and complete
neural-codec performance remain unverified.

Validation passed 5,959,833 reader checks, 24 decoder cases, four real geometry
streams and 447 original arithmetic compatibility cases. The full pytest run
reported 87 passed, 49 subtests passed and three CUDA-dependent skips. The
Morton XYZ batching trial changed total real-cloud hierarchy latency from
161.1505 to 162.3079 ms without a stable improvement. CDF cache blocking also
showed no stable gain in its separate CDF benchmark. Both changes were reverted.

Raw samples, compiler/environment details, source hashes and validation are in
[the retained benchmark](../benchmarks/results/memory_io_20261020/bitreader_final.json),
[the experiment summary](../benchmarks/results/memory_io_20261020/experiment_summary.json)
and [the pytest log](../benchmarks/results/memory_io_20261020/tests_final.txt).
The report includes a [benchmark JSON copy](memory_io_benchmark.json) and
[CSV comparison](memory_io_benchmark.csv). Reproduce from the repository root,
writing a separate result to preserve the original measurements:

```powershell
& 'C:\Users\LinkZelda\miniconda3\python.exe' benchmarks/bench_bitreader_batch.py --before benchmarks/results/memory_io_20261020/before/csrc/arithmetic_core.h --benchmark --geometry-histograms --output benchmarks/results/memory_io_20261020/bitreader_retest.json
```

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
