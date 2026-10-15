# ELiCv1 优化、实测与验收报告

日期：2026-10-15（Asia/Hong_Kong）。原始版本：`8f3ea78a73d4a3b1ac8d58912fcca10523e4d4db`。

已实现并验证的 CPU 模块中，多项达到数倍至数十倍加速。完整神经网络编解码和 GPU 整机加速仍待验证：本机 CUDA 不可用，缺少 TorchSparse；生产 PyTorch C++ 绑定还因 Windows SDK/UCRT `assert.h` 缺失而无法编译。不能将以下 CPU 模块倍数作为 ELiCv1 整机倍数，也不能据此宣称完整神经 codec 已验收。

本次全部可运行回归：**87 项通过、49 个子测试通过、3 项 CUDA 测试跳过，0 失败**。详见 [测试输出](test_output.txt) 和 [JUnit](tests.xml)。完整模型预检结果为 **unavailable**，生产绑定构建结果为 **unavailable**，分别保存在 [模型验收状态](pipeline_validation.json) 和 [绑定构建状态](arithmetic_torch_binding.json)。

## 整体优化结果

本轮在四份真实点云、共 412,378 个输入点上，完整几何层级构建的平均耗时由 **295.522 ms/帧降至 22.094 ms/帧**，汇总加速 **13.376×**。BoE 离线完整直方图收集汇总加速 **13.618×**；PLY 读写合计和经验 CDF 的整数算术编解码合计分别为 **2.240×**、**1.482×**。

以下按完整子流程汇总已有实测记录。先对四帧各分项的计时中位数求和，再除以帧数得到平均耗时；加速比为原版总耗时除以优化总耗时，未对各帧加速比取算术平均。读写、算术编解码由各自分项中位数相加，**不是将这些步骤连起来重新计时的端到端实验**。

| 完整子流程 | 原版平均 ms/帧 | 优化平均 ms/帧 | 汇总加速 | 耗时下降 |
| --- | ---: | ---: | ---: | ---: |
| 完整几何层级构建 | 295.522 | 22.094 | 13.376× | 92.52% |
| BoE 离线完整直方图收集 | 360.086 | 26.443 | 13.618× | 92.66% |
| PLY 读取＋写出 | 523.359 | 233.607 | 2.240× | 55.36% |
| 整数算术编码＋解码 | 29.641 | 20.002 | 1.482× | 32.52% |

### 已测 CPU 阶段成本合计（分析值）

仅将「几何层级构建＋单文件 PLY 读取/写出＋经验 CDF 算术编码/解码」相加，四帧成本由 **3394.0888 ms → 1102.8118 ms**，折合平均 **848.522 → 275.703 ms/帧**，成本比为 **3.078×**，下降 **67.51%**。该数值只描述这些已测 CPU 工作负载的分析性成本预算，**不是 ELiCv1 完整神经编解码的整体加速比**。

计算只使用完整层级构建一行，不再叠加其排序、位交织和下采样；I/O 只使用每帧读取和写出，不再累计批量读取行。BoE 属于独立的离线训练前处理，且 posQ=4、内部也包含层级构建，因此不并入上述 posQ=16 的成本合计。单模块上采样和合成 CDF 基准没有覆盖完整真实链路的调用次数，也未混入合计。

### 完整 ELiCv1 的结论与待验收指标

**当前能确认的是已测 CPU 子流程明显提速，以及现有可运行的模块/几何/整数码流测试通过；完整神经编解码能否整体达到若干倍，仍无实测结论。** 上述分析合计未包含稀疏卷积与预测、完整解码逐层上采样、实际神经 CDF 生成/归一化、CPU/GPU 传输与同步、模型/JIT 启动等成本，不能据此推导真实 FPS、整机延迟或 GPU 加速比。I/O 写出基准使用原样例坐标，实际解码输出的去重点数也可能不同。

模型架构、权重及预测精度保持原样；完整模型的压缩率、历史神经码流双向兼容、总峰值内存和训练收敛仍待相应环境验证。Windows SDK/UCRT、CUDA 与 TorchSparse 就绪后，使用 `benchmarks/compare_pipeline.py` 对原新版本做完整编解码、逐字节及量化坐标验收，分别报告编码、解码、含读写的整条流程耗时和异步模式收益。

整体汇总数值与计算口径保存于 [JSON](overall_summary.json) 和 [CSV](overall_summary.csv)，可通过 `python benchmarks/write_report.py` 从已有原始记录重新生成，无需重新计时。

## 逐模块实测

| 模块 | 原版耗时 / 优化耗时 | 数据与范围 |
| --- | ---: | --- |
| Morton 位交织编码 | 1.18–1.55× | 四份真实样例，posQ=16 |
| 初始 Morton 排序与去重 | 24.58–29.14× | 四份真实样例，posQ=16 |
| 单层下采样和占用符号 | 1.36–1.68× | 四份真实样例，posQ=16 |
| 完整几何层级构建 | 12.25–15.24× | 四份真实样例，posQ=16 |
| 特征上采样 | 4.61–7.07× | 四份真实样例，posQ=16 |
| 坐标上采样 | 3.29–3.86× | 四份真实样例，posQ=16 |
| 坐标和特征联合上采样 | 4.85–6.21× | 四份真实样例，posQ=16 |
| BoE 离线直方图统计 | 52.25–61.80× | 中心训练前处理，posQ=4 |
| BoE 离线完整直方图收集 | 12.18–14.84× | 中心训练前处理，posQ=4 |
| PLY 读取 | 3.72–4.93× | 四份真实样例；缓存读写 |
| PLY 写出 | 1.49–1.52× | 四份真实样例；缓存读写 |
| 整数算术编码 | 1.37–2.01× | 真实占用符号＋经验 CDF，非神经预测 |
| 整数算术解码 | 1.30–1.42× | 真实占用符号＋经验 CDF，非神经预测 |
| CDF 归一化（N=64） | 1.55× | 17 项 CDF；合成数据 |
| CDF 归一化（N=4096） | 1.09× | 17 项 CDF；合成数据 |
| CDF 归一化（N=100000） | 1.09× | 17 项 CDF；合成数据 |
| 推理时 BoE 选择、稀疏卷积、Embedding、预测网络 | 未测，主要计算保持原样 | 需 CUDA/TorchSparse 完整验收 |
| GPU/CPU 异步熵编码流水 | GPU 收益未测 | 可选 `--async_entropy True`；真实 C++ 核线程顺序测试通过 |

四份样例合计 **412,378** 个输入点。完整层级构建包含初始排序和多层下采样，BoE 完整收集包含其层级构建；这些收益存在重叠，**不能相加或相乘**。BoE 离线前处理结果不代表推理时选择编码器的加速，KMeans、TSNE 未计时、未改变。

算术核在四份经验占用流上的合计耗时比为编码 **1.655×**、解码 **1.369×**。极端单峰合成分布的编码为 **0.945×**，约慢 **5.9%**；该回退保留披露。CDF 小模块、不同分布及系统负载都会影响倍数，原始计时样本均保存在 JSON。

## 实现与论文对应

1. `src/morton.py`：先按 Morton 整数排序，再相邻去重，去掉二维 unique 的额外排序；位扩张复用临时存储。单层下采样直接用 `code & 7` 得到子八分体，仅对父结点提取坐标。`build_hierarchy` 只读取一次最大位深，去掉逐层主机同步及无用克隆。
2. 特征和坐标上采样按实际占用的 parent/child 索引 gather，避免实体化 `8N×C` 特征和所有候选坐标；联合上采样共享索引。保持 Morton 顺序、坐标与特征对齐；训练特征梯度保留。
3. `src/op.py`：用精确可表示的常量代替每次构造标量 tensor/pow，按 device 缓存 CDF 位置增量；保持 round、int16 和末项 65536 的原约定。
4. `csrc/arithmetic_core.h`：区间状态使用局部变量，CLZ 批量处理 E1/E2/E3 与 bit I/O，常规 CDF 用四次乘法比较替代逐符号除法；极偏 CDF 保留原搜索。`arithmetic_coding.cpp` 保持 API，并在纯 Tensor Chunk 调用时释放 GIL。
5. `src/runtime.py`、`compress.py`：两阶段 CDF/符号按原顺序合并为一次 Chunk，减少传输/调用；单工作线程 FIFO 可选重叠 CPU 算术编码和下一层 GPU 预测，finish 等待后重置，限制队列长度。默认关闭异步，等待 GPU 验收。ELiC PDF 第 13–14 页补充 E 节报告的是论文自身约 1.142×，并非本机结果。
6. `src/io.py`：按 PLY header 只读 vertex 的命名 XYZ 列，NumPy 批量解析/写出，移除无用 Open3D 对象；有限线程替代固定 64 进程。写出直接覆盖，修复 Windows 下删除失败后追加重复 PLY 的问题。输出以 float32 的 9 位有效数字保值，ASCII 文本/文件大小可能变化。
7. `src/boe.py`、`BoE_cluster.py`：统计占用直方图无需空间排序，移除每层四次排序，复用新层级构建；保持逐层 32 bin 数值、位深标签与样本截断顺序。
8. 驱动脚本增加 main guard、明确 CUDA 前提、正确处理布尔 False、空 glob、独立解压 CSV、1–2 帧均值；编解码计时采用同步 wall clock，修复原解码 `[1:-1]` 得到 NaN 的统计问题。模型结构、checkpoint、神经精度和二进制格式保持原样。

ELiC 的 Morton/跨位深传播已是原项目算法，不能把论文相对其他方法的收益再算成本次提升。Tetris 研究的是 2D CNN 的权重稀疏，其加速倍数不能转用于 ELiC 的 3D 空间稀疏。本次借鉴其减少中间物化与访存的思路；未声称实现 Tetris CUDA kernel。论文页码及适用边界见 [论文笔记](paper_notes.md)。

## 正确性验证覆盖与限制

| 范围 | 已执行的验证 |
| --- | --- |
| Morton/占用/传播 | 与冻结原版逐元素比较；全部 256 占用符号、21 位边界、重复/空输入、随机多层重建、32/64 通道、显式多 batch 后备模式。前向精确一致；double 梯度因累加顺序不同采用 1e-14 容差。codec 仍使用原来的 batch=1。 |
| CDF | float16/32/64、17 项及其他长度、零概率、边界与半整数取整，int16 精确相等。测试 float16 转换不表示模型启用 FP16。 |
| 算术核 | 447 组随机/边界、全部符号、最小频率、长 E3 pending、不同 Chunk 切分、中间 buffer、字节消耗、默认及非默认 reset/finish、空流；编码字节相同、旧新双向互解。 |
| 真实几何＋整数熵编码 | 4 份完整样例 × 5 档 posQ=4/8/16/32/64，逐层占用/坐标精确一致；原新编码、四种编解码组合与量化去重目标一致。此项采用由已解坐标和阶段决定的确定性 CDF，**不含训练后神经预测**。 |
| BoE | 冻结原脚本循环 AST 对照，逐层 32 bin、位深标签、跨帧截断顺序、冻结模型的中心选择及平局处理一致。未重新聚类或训练。 |
| I/O | 4 份实际点云原新读入精确一致；额外属性/列重排/其他元素/空单点/损坏与截断/BIN；包含负零、非规格数的 float32 回读；覆盖写出、空格路径与线程顺序。 |
| 异步 | 真实 C++ 整数核经 ctypes 接入生产 EntropyEncoder，分块 FIFO 字节一致、重复 finish/reset；工作线程错误传播。**GPU 重叠和生产 pybind 编译/运行未验证**。 |
| 完整模型与训练 | CUDA/TorchSparse 不可用，未完成真实神经 CDF、原新完整码流双向互解、D1/D2 外部程序、训练收敛或 GPU 端到端时延验收。 |

## 性能测试方法与复现

环境：Windows 11，Intel Family 6 Model 170，22 个逻辑处理器；Python 3.13.12、PyTorch 2.12.0+cpu、NumPy 2.4.2。几何/CDF/BoE 固定 **1 线程**，原版和优化版相同；I/O 默认最多 min(文件数, CPU 数, 8) 个线程，另有顺序对照。基准依次独占运行，预热 2–3 次，交替原新计时 7/9/11 次，取中位数，定义加速比为 `median(T_original)/median(T_optimized)`。这些范围是四个样例的观测范围，非置信区间。

几何与 CDF 比较只计模块调用，输入预先装载；I/O 计打开/解析或格式化/关闭，使用系统缓存、不做 fsync，不代表冷盘速度。原始 occupancy lookup 表只迁移到 CPU，保留其原计算。原 I/O 基线移除了无用 Open3D 构造；批量基线为原单文件读取的顺序调用，因为原 Pool(64) 的 Windows 驱动不可直接运行。算术基线由冻结源码提取，仅替换 Torch/Python 容器接口并去掉 Torch 校验宏，优化核仍检查符号范围；这有利于基线，并未夸大收益。

源码与权重哈希见 [验证清单](validation_manifest.json)。所有模块和样例的原版毫秒、优化毫秒、倍数集中在 [CSV](module_speedups.csv)。原始记录分别为 [几何](../benchmarks/results_morton.json)、[BoE](boe_cpu.json)、[I/O](io_benchmark.json)、[CDF/合成](modules_cpu.json)、[算术核](arithmetic_benchmark.json)。

完整命令与输入单位说明见 [复现指南](../OPTIMIZATION.md)。重点入口：

```powershell
$elicPython = 'C:\Users\LinkZelda\miniconda3\python.exe'
& $elicPython -m pytest -q --junitxml=reports/tests.xml
& $elicPython benchmarks/bench_morton.py
& $elicPython benchmarks/bench_io.py --repeats 7
& $elicPython benchmarks/bench_boe.py
& $elicPython benchmarks/bench_modules.py --device cpu --threads 1 --repeats 11
& $elicPython benchmarks/benchmark_arithmetic.py --benchmark --geometry-histograms --output reports/arithmetic_benchmark.json
```

在可运行原模型的 CUDA/TorchSparse 环境，执行 `python benchmarks/compare_pipeline.py --repeats 3`，再加 `--async_entropy --output reports/pipeline_async.json` 单独验收异步。脚本覆盖两个数据集和全部五档 posQ，逐字节比较压缩文件、双向交叉解码并比较量化去重的坐标集合；各轮新输出文件，预热轮不计统计，保留日志及逐轮耗时。原版驱动仅在临时副本加入 main guard 以兼容 Windows spawn，计算算法不变。

GPU 验收通过后才可报告整机倍数。若瓶颈仍在稀疏卷积，应先测 kernel map、gather/scatter、矩阵计算和同步各自占比，再决定 CUDA kernel 优化；当前环境不能支持可靠的卷积实现选择。模型精度、剪枝或重训练会影响 CDF/旧流兼容性，未纳入本轮兼容优化。

## 全部样例耗时明细

几何/BoE 的完整收集行包含子模块，不能重复计入总收益。`synthetic_100k_cpu` 的 CDF 行在模块名中另标明实际 N=64/4096。

| 范围 | 模块 | 样例 | 原版 ms | 优化 ms | 加速 |
| --- | --- | --- | ---: | ---: | ---: |
| real_geometry_cpu | morton3_code | data/FordExample/sample_0001.ply | 1.4995 | 0.9677 | 1.550× |
| real_geometry_cpu | morton3_sort | data/FordExample/sample_0001.ply | 264.7000 | 9.0835 | 29.141× |
| real_geometry_cpu | down_once | data/FordExample/sample_0001.ply | 2.4962 | 1.7920 | 1.393× |
| real_geometry_cpu | build_hierarchy | data/FordExample/sample_0001.ply | 243.3267 | 19.5006 | 12.478× |
| real_geometry_cpu | upscale_feature | data/FordExample/sample_0001.ply | 28.8364 | 4.0815 | 7.065× |
| real_geometry_cpu | upscale_coordinate | data/FordExample/sample_0001.ply | 18.0881 | 4.6806 | 3.864× |
| real_geometry_cpu | upscale_coordinate_feature | data/FordExample/sample_0001.ply | 34.5350 | 5.7088 | 6.049× |
| real_geometry_cpu | morton3_code | data/FordExample/sample_0002.ply | 1.1422 | 0.9709 | 1.176× |
| real_geometry_cpu | morton3_sort | data/FordExample/sample_0002.ply | 204.9070 | 7.7092 | 26.580× |
| real_geometry_cpu | down_once | data/FordExample/sample_0002.ply | 2.5424 | 1.8692 | 1.360× |
| real_geometry_cpu | build_hierarchy | data/FordExample/sample_0002.ply | 223.0855 | 18.2054 | 12.254× |
| real_geometry_cpu | upscale_feature | data/FordExample/sample_0002.ply | 20.3137 | 4.3751 | 4.643× |
| real_geometry_cpu | upscale_coordinate | data/FordExample/sample_0002.ply | 17.2508 | 5.2394 | 3.293× |
| real_geometry_cpu | upscale_coordinate_feature | data/FordExample/sample_0002.ply | 36.1126 | 5.8120 | 6.213× |
| real_geometry_cpu | morton3_code | data/SemanticKITTIExample/sample_0001.ply | 2.4172 | 1.7280 | 1.399× |
| real_geometry_cpu | morton3_sort | data/SemanticKITTIExample/sample_0001.ply | 348.8575 | 14.1951 | 24.576× |
| real_geometry_cpu | down_once | data/SemanticKITTIExample/sample_0001.ply | 5.1959 | 3.0919 | 1.680× |
| real_geometry_cpu | build_hierarchy | data/SemanticKITTIExample/sample_0001.ply | 370.9272 | 28.0469 | 13.225× |
| real_geometry_cpu | upscale_feature | data/SemanticKITTIExample/sample_0001.ply | 29.4581 | 6.0815 | 4.844× |
| real_geometry_cpu | upscale_coordinate | data/SemanticKITTIExample/sample_0001.ply | 27.7848 | 7.6004 | 3.656× |
| real_geometry_cpu | upscale_coordinate_feature | data/SemanticKITTIExample/sample_0001.ply | 60.5973 | 10.5034 | 5.769× |
| real_geometry_cpu | morton3_code | data/SemanticKITTIExample/sample_0002.ply | 2.4144 | 1.9113 | 1.263× |
| real_geometry_cpu | morton3_sort | data/SemanticKITTIExample/sample_0002.ply | 343.1399 | 13.4941 | 25.429× |
| real_geometry_cpu | down_once | data/SemanticKITTIExample/sample_0002.ply | 4.5087 | 2.9659 | 1.520× |
| real_geometry_cpu | build_hierarchy | data/SemanticKITTIExample/sample_0002.ply | 344.7493 | 22.6229 | 15.239× |
| real_geometry_cpu | upscale_feature | data/SemanticKITTIExample/sample_0002.ply | 28.2050 | 6.1169 | 4.611× |
| real_geometry_cpu | upscale_coordinate | data/SemanticKITTIExample/sample_0002.ply | 26.8551 | 7.4079 | 3.625× |
| real_geometry_cpu | upscale_coordinate_feature | data/SemanticKITTIExample/sample_0002.ply | 58.0460 | 11.9618 | 4.853× |
| offline_boe_cpu | boe_offline_sort_histograms | data/FordExample/sample_0001.ply | 68.7839 | 1.2011 | 57.267× |
| offline_boe_cpu | boe_offline_collect | data/FordExample/sample_0001.ply | 275.0657 | 22.5817 | 12.181× |
| offline_boe_cpu | boe_offline_sort_histograms | data/FordExample/sample_0002.ply | 71.0140 | 1.1491 | 61.800× |
| offline_boe_cpu | boe_offline_collect | data/FordExample/sample_0002.ply | 278.9513 | 21.1375 | 13.197× |
| offline_boe_cpu | boe_offline_sort_histograms | data/SemanticKITTIExample/sample_0001.ply | 93.9990 | 1.7991 | 52.248× |
| offline_boe_cpu | boe_offline_collect | data/SemanticKITTIExample/sample_0001.ply | 429.7621 | 28.9670 | 14.836× |
| offline_boe_cpu | boe_offline_sort_histograms | data/SemanticKITTIExample/sample_0002.ply | 102.2419 | 1.6649 | 61.410× |
| offline_boe_cpu | boe_offline_collect | data/SemanticKITTIExample/sample_0002.ply | 456.5659 | 33.0851 | 13.800× |
| io_cpu | PLY read | data/FordExample/sample_0001.ply | 157.8986 | 38.8092 | 4.069× |
| io_cpu | PLY write | data/FordExample/sample_0001.ply | 198.4799 | 131.8874 | 1.505× |
| io_cpu | PLY read | data/FordExample/sample_0002.ply | 175.5721 | 47.2337 | 3.717× |
| io_cpu | PLY write | data/FordExample/sample_0002.ply | 200.8578 | 132.5201 | 1.516× |
| io_cpu | PLY read | data/SemanticKITTIExample/sample_0001.ply | 334.9477 | 67.9074 | 4.932× |
| io_cpu | PLY write | data/SemanticKITTIExample/sample_0001.ply | 310.6554 | 206.0988 | 1.507× |
| io_cpu | PLY read | data/SemanticKITTIExample/sample_0002.ply | 368.3512 | 77.6577 | 4.743× |
| io_cpu | PLY write | data/SemanticKITTIExample/sample_0002.ply | 346.6742 | 232.3123 | 1.492× |
| io_cpu | Batch read sequential | all four real samples | 1082.8711 | 238.0885 | 4.548× |
| io_cpu | Batch read auto | all four real samples | 1036.6882 | 230.5008 | 4.498× |
| synthetic_100k_cpu | morton_sort | 100000 | 335.9527 | 12.9576 | 25.927× |
| synthetic_100k_cpu | down_once | 100000 | 3.7281 | 2.5812 | 1.444× |
| synthetic_100k_cpu | upscale_feature | 100000 | 28.7324 | 6.2072 | 4.629× |
| synthetic_100k_cpu | upscale_coordinate | 100000 | 24.0740 | 6.1178 | 3.935× |
| synthetic_100k_cpu | upscale_coordinate_feature | 100000 | 50.1365 | 8.2864 | 6.050× |
| synthetic_100k_cpu | hierarchy | 100000 | 338.9861 | 34.5755 | 9.804× |
| synthetic_100k_cpu | cdf_normalization | 100000 | 3.0152 | 2.7594 | 1.093× |
| synthetic_100k_cpu | cdf_normalization_64 | 100000 | 0.0295 | 0.0190 | 1.553× |
| synthetic_100k_cpu | cdf_normalization_4096 | 100000 | 0.1412 | 0.1301 | 1.085× |
| arithmetic_integer_core_cpu | arithmetic_encode | uniform | 10.5185 | 3.0377 | 3.463× |
| arithmetic_integer_core_cpu | arithmetic_decode | uniform | 13.9099 | 9.7727 | 1.423× |
| arithmetic_integer_core_cpu | arithmetic_encode | concentrated | 1.4818 | 1.5687 | 0.945× |
| arithmetic_integer_core_cpu | arithmetic_decode | concentrated | 3.8841 | 3.4046 | 1.141× |
| arithmetic_integer_core_cpu | arithmetic_encode | random_weighted | 15.2054 | 9.0521 | 1.680× |
| arithmetic_integer_core_cpu | arithmetic_decode | random_weighted | 23.9978 | 13.2947 | 1.805× |
| arithmetic_integer_core_cpu | arithmetic_encode | random_rare_symbols | 17.1286 | 8.5579 | 2.002× |
| arithmetic_integer_core_cpu | arithmetic_decode | random_rare_symbols | 27.3175 | 13.4766 | 2.027× |
| arithmetic_integer_core_cpu | arithmetic_encode | histogram_Ford_sample_0001 | 11.6498 | 8.4965 | 1.371× |
| arithmetic_integer_core_cpu | arithmetic_decode | histogram_Ford_sample_0001 | 16.0535 | 11.8466 | 1.355× |
| arithmetic_integer_core_cpu | arithmetic_encode | histogram_Ford_sample_0002 | 14.1215 | 7.0391 | 2.006× |
| arithmetic_integer_core_cpu | arithmetic_decode | histogram_Ford_sample_0002 | 16.6462 | 12.8541 | 1.295× |
| arithmetic_integer_core_cpu | arithmetic_encode | histogram_SemanticKITTI_sample_0001 | 12.4062 | 7.8315 | 1.584× |
| arithmetic_integer_core_cpu | arithmetic_decode | histogram_SemanticKITTI_sample_0001 | 16.7917 | 11.8196 | 1.421× |
| arithmetic_integer_core_cpu | arithmetic_encode | histogram_SemanticKITTI_sample_0002 | 13.9997 | 8.1619 | 1.715× |
| arithmetic_integer_core_cpu | arithmetic_decode | histogram_SemanticKITTI_sample_0002 | 16.8946 | 11.9601 | 1.413× |
