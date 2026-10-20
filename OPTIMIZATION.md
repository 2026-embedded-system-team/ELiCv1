# ELiCv1 优化与复现

详细结果见 [模块优化报告](reports/optimization_report.md)，论文依据和页码见 [论文笔记](reports/paper_notes.md)。

本机为 Windows + CPU PyTorch，没有 CUDA/TorchSparse。CPU 模块加速和几何/整数熵编码的验证结果不能代替完整神经网络的 GPU 验收。原模型结构、checkpoint、预测精度和二进制格式保持原样；异步编码通过 `--async_entropy True` 显式开启。

2026-10-20 新增访存优化：`BitReader` 用有界 `memcpy` 一次读取相邻 4 字节，放入 64 位位缓存，保留尾部补零和逻辑已消费字节数。以本次修改前的优化版本为基线，预热 3 次、交替测试 15 轮；四份 Ford/SemanticKITTI 完整样例（posQ=16、逐层经验 CDF）的 CPU 整数解码中位数合计从 **48.9337 ms 降至 42.7891 ms**，即 **1.1436×、耗时下降 12.56%**。均匀合成输入为 0.993×（约慢 0.75%），高度集中输入为 1.005×，没有明确收益。该增量比较单独保留，不与此前基于冻结原版的结果相乘或合并，也不代表 GPU/完整模型加速。Morton/CDF 批处理试验没有稳定收益，已撤回。

本次验证为 87 项测试及 49 项子测试通过、3 项 CUDA 测试跳过；另有 5,959,833 项读位检查、24 个解码用例、4 份真实样例和 447 项原版码流兼容性检查通过。方法、逐样例结果和复现说明见 [算术编码补充说明](reports/arithmetic_notes.md#incremental-memory-io-update-2026-10-20)，数据见 [报告 JSON](reports/memory_io_benchmark.json)、[CSV](reports/memory_io_benchmark.csv) 和 [原始测量](benchmarks/results/memory_io_20261020/bitreader_final.json)。

在仓库根目录执行，使用具有 PyTorch、NumPy、pytest、pandas 的 Python；独立算术核测试还需要 g++ 或 clang++。

```powershell
# 本次实际使用的 Python（默认 python 没有安装 PyTorch）
$elicPython = 'C:\Users\LinkZelda\miniconda3\python.exe'
& $elicPython -m pytest -q --junitxml=reports/tests.xml
& $elicPython benchmarks/bench_morton.py
& $elicPython benchmarks/bench_io.py --repeats 7
& $elicPython benchmarks/bench_modules.py --device cpu --threads 1 --repeats 11
& $elicPython benchmarks/bench_boe.py
& $elicPython benchmarks/benchmark_arithmetic.py --benchmark --geometry-histograms --output reports/arithmetic_benchmark.json
# 单独复测 2026-10-20 的访存增量，保留原始测量文件
& $elicPython benchmarks/bench_bitreader_batch.py --before benchmarks/results/memory_io_20261020/before/csrc/arithmetic_core.h --benchmark --geometry-histograms --output benchmarks/results/memory_io_20261020/bitreader_retest.json
```

性能基准逐个运行，避免互相争用 CPU。使用原始与优化版本交替计时的中位数，原始样本存入 JSON。`benchmarks/reference` 冻结原始源码，`manifest.json` 记录版本和 SHA-256。`pytest.ini` 将临时测试文件放在工作区内。

CUDA 机器上需沿用能运行原项目的 PyTorch + TorchSparse 环境，并具备原版 Open3D、pandas、tqdm、JIT 所需的 ninja 和 C++ 编译器。先运行模块验收，再测试完整模型：

```bash
python -m pytest -q --junitxml=reports/tests_cuda.xml
python benchmarks/bench_morton.py --device cuda
python benchmarks/bench_modules.py --device cuda --threads 1 --output reports/modules_cuda.json
python benchmarks/compare_pipeline.py --repeats 3
python benchmarks/compare_pipeline.py --repeats 3 --async_entropy --output reports/pipeline_async.json
```

完整验收覆盖 Ford/SemanticKITTI、posQ=4/8/16/32/64、四份完整样例、原新编码字节比较、双向交叉解码、量化去重后的 XYZ 集合精确比较。包含一次不计入统计的编译/缓存预热；正式轮次交替运行原版和优化版，各轮使用新输出文件。记录逐帧 codec 延迟与包含启动/读写的进程总耗时。缺少 GPU/依赖时明确生成 `unavailable` 结果并以状态码 2 退出，不会报成功。

手工测试 SemanticKITTI：

```bash
python compress.py --input_glob './data/SemanticKITTIExample/*.ply' --output_folder ./data/SemanticKITTI/ --posQ 16
python decompress.py --input_glob './data/SemanticKITTI/*.bin' --output_folder ./data/SemanticKITTI/
```

Ford 样例使用毫米整数，压缩与解压都加 `--is_data_pre_quantized True --ckpt ./model/Ford/ELiCv1_K5.pt`，并使用 Ford 输入/输出目录。SemanticKITTI 样例使用米浮点，保持默认 False。两者不能混用量化选项。

`eval.py`/`third_party/pc_error_d` 是原项目的外部 D1/D2 指标流程，未在本机执行。上述精确坐标验收直接验证量化后几何保持一致；GPU 验收通过后可在具备该可执行文件的环境沿用原有 PSNR 评估。
