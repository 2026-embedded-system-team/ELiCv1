# ELiCv1 优化与复现

详细结果见 [模块优化报告](reports/optimization_report.md)，论文依据和页码见 [论文笔记](reports/paper_notes.md)。

本机为 Windows + CPU PyTorch，没有 CUDA/TorchSparse。CPU 模块加速和几何/整数熵编码的验证结果不能代替完整神经网络的 GPU 验收。原模型结构、checkpoint、预测精度和二进制格式保持原样；异步编码通过 `--async_entropy True` 显式开启。

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
