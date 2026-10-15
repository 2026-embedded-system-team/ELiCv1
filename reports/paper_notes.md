# ELiCv1 论文依据与优化边界

阅读来源为工作区提供的 PDF。下文页码均为 PDF 文件页码，从 1 开始；`3627535.3638471-final.pdf` 的印刷页码比 PDF 页码大 228。

## 1. ELiC

**题目**：ELiC: Efficient LiDAR Geometry Compression via Cross-Bit-depth Feature Propagation and Bag-of-Encoders。Junsik Kim、Gun Bang、Soowoong Kim；arXiv:2511.14070v3，2026-04-30。

- 第 3–4、12–13 页：一次 Morton 排序后，按父结点分组下采样，按父结点顺序和 Morton 八叉子结点顺序上采样，可以一直保持全局顺序。坐标、占据符号、传播特征的顺序必须一致。
- 第 6 页：在相同输入位深下，对量化输入坐标执行无损编码。测试应与量化、去重后的坐标集合比较，而非要求恢复量化前实数或重复点。
- 第 7–8 页，Table 1 / Table 5：论文中 Morton 方案已包含在 ELiC 基线。显式逐层排序的平均编码/解码时间 0.142 / 0.128 秒，Morton 方案 0.121 / 0.111 秒。此项不能再次计为相对当前源码的新收益。
- 第 13–14 页，补充材料 E 节、Table S2：GPU 在计算下一位深时，由 CPU 线程对上一位深做算术编码。论文明确此调整不改变码流或压缩效率。ELiC 平均编码时间 0.121 → 0.106 秒，约 **1.142×**；这是论文结果，不是本机实测。

| 输入位深 | ELiC 同步编码，秒 | 异步编码，秒 | 论文中的加速比 |
| --- | ---: | ---: | ---: |
| 16 | 0.172 | 0.145 | 1.186× |
| 15 | 0.145 | 0.125 | 1.160× |
| 14 | 0.121 | 0.105 | 1.152× |
| 13 | 0.095 | 0.086 | 1.105× |
| 12 | 0.074 | 0.068 | 1.088× |

这些论文实验使用 RTX 3090、i9-9900K、64 GB RAM，WSL2。不能将其数值直接作为当前电脑的测试结果。

## 2. Tetris

**题目**：Tetris: Accelerating Sparse Convolution by Exploiting Memory Reuse on GPU。Xiaoyan Liu、Xuegui Zheng、Hailong Yang、Zhongzhi Luan、Depei Qian；PPoPP 2024；DOI:10.1145/3627535.3638471。

- 第 3 页，§2.1–2.3：研究对象是剪枝后卷积权重稀疏的 2D CNN；输入特征图稠密。ELiC 的 TorchSparse 则是 3D 空间坐标稀疏，卷积权重通常稠密，两者并非可直接替换的算子。
- 第 3–4 页，§2.3 / §3.1：在卷积空间位置、输出通道及输入通道之间复用输入数据，减少全局内存访问；通过共享内存和寄存器分块实现。论文中的“3D tiling”指三个复用方向，并非直接针对 3D 点云卷积。
- 第 4–5 页，§3.2：SPF 按 warp 打包稀疏权重，支持合并和向量访存。直接迁入 ELiC 需要先建立真实权重稀疏性，且可能涉及剪枝/重训练，因此不适合作为默认、兼容旧 checkpoint 的修改。
- 第 5–7 页，§3.3–3.4：bank 冲突重排、负载均衡以及根据 GPU 资源和测量选择分块参数，适合作为自定义 CUDA 空间稀疏算子的设计参考。需要有实际 GPU profiler 结果支持，不能仅靠 Python 重写声称实现。
- 第 8 页，§4.3：全部卷积核相对 cuDNN / TACO-UCF / cuSPARSE / Sputnik 的几何平均加速为 1.93× / 4.15× / 2.09× / 1.60×。这是论文所测权重稀疏卷积核的结果，不能转用为 ELiC 或整条编解码链的加速。
- 第 8–9 页：小尺寸、1×1、低 batch 和过高稀疏可能缺少复用或并行度。ELiC 低位深点数少，应特别测量算子启动、索引和同步开销。

## 3. 与当前代码的对应

以下为源码审查提出的候选优化，不代表已实现或已测得收益。

| 模块 | 源码现状 | 可保持功能的候选优化 | 核心验证 |
| --- | --- | --- | --- |
| Morton 初始排序 | `torch.unique(bxyz, dim=0)` 后另算 Morton 并排序 | 在坐标范围和 batch=1 契约内，先 Morton 排序，再相邻去重；避免二维 unique 的另一次排序 | 原、新坐标和 Morton code 逐元素一致；高坐标/重复点/空输入边界 |
| 层级下采样 | 每层重复坐标 LSB 运算、inverse、scatter、clone | 复用 `code & 7` 表示 octant；仅在需要时生成 inverse；减少无用 clone；起始最大位深只同步一次 | 所有层占据和顺序完全相同；不能相加重复 child 占据位 |
| 特征上采样 | 先复制 `8N × C` 再以 mask 丢弃未占据项 | 由占据 popcount 或 mask 的非零位置得到 parent index，直接 gather 实际 child 特征 | 全部 256 个占据符号，32/64 通道，值和梯度一致 |
| 坐标和特征联合上采样 | 坐标、偏移和特征都先 8 倍扩张 | 只对有效 parent/child 索引计算 `2*xyz + octant`；联合复用索引 | Morton 顺序、坐标与特征对齐逐元素相等 |
| 位深及 BoE | 每层 `.max().item()`，histogram/index 同步 | 首层确定位深后随层递减；考虑批量准备层选择，但保持相同距离运算及 tie 行为 | 编码器索引列表相同；特别测试相等中心和边界输入 |
| CDF 归一化 | 每次创建标量 tensor、`pow_(16)`、arange，多次中间分配 | 16-bit 常量和位置增量按 device/dtype 复用，减少临时数组 | 17 项 int16 CDF 精确相同，包含末项 unsigned 65536 的特殊约定 |
| 算术编码 | 每层 GPU→CPU 后，主线程同步编两个阶段 | 有序 producer/consumer；CPU buffer 和 CUDA event 保证数据已可读 | 完整 bitstream 字节一致；跨多 chunk 状态一致 |
| C++/Python 并行 | `encode_chunk` pybind 未释放 GIL | 在纯张量 `encode_chunk` 绑定释放 GIL；`finish` 生成 `py::bytes` 时仍需 GIL | 防止伪并行；同一个 encoder 保证只有一个消费者 |
| 文件读取 | 固定 `Pool(64)`；脚本顶层启动池 | Windows 使用 `__main__`，小输入顺序读取，worker 随 CPU 和文件数确定 | 默认样例 CLI、单文件和多文件读入相同 |
| PLY 写出 | 无用 open3d 对象；逐点多次 Python 写；先 shell `rm` 后 append | 直接 `'w'` 覆盖；批量字符串/NumPy 块写 | 重读坐标等价，重跑不重复 header，路径空格可用 |

不能在默认兼容模式中擅自启用 FP16、TF32、量化、剪枝、改变矩阵归约次序或 BoE 选择度量。微小概率差异可能改变量化后的整数 CDF，导致历史码流解码失败。浮点 `allclose` 不能代替 CDF 精确相等和交叉解码。

原始实现的 `morton3_sort` 虽有 batch 列，`down_once` 将 batch 设为 0，模型说明也仅支持 batch size=1。不要把非支持的多 batch 行为当作优化收益；如扩展支持，应独立声明并测试。

## 4. 性能与完整性测试方法

1. 保存原始模块/原始可执行入口，固定同一输入、同一 checkpoint、设备、线程数和精度设置。
2. 对整数几何、占据、CDF、输出 bytes 使用精确比较；浮点中间特征增加差值检查，但最终仍要求 CDF/码流以及重建一致。
3. 构造随机点云、重复点、全部 256 种占据符号、单一子结点、满八叉、极稀疏、32/64 通道、低位深和高位深边界；几何测试还要验证按顺序逐层 down/up 重建。
4. 实际样例上比较：原始编码→原始解码，优化编码→优化解码，原始编码→优化解码，优化编码→原始解码。四者的量化坐标集合应一致；兼容优化要求原、新压缩 bytes 一致。
5. 模块 benchmark 应预热、固定输入、交替 baseline/optimized，报告中位数及至少 p10/p90；先记录 CPU 线程数，GPU 计时前后同步。不能把 CPU 实测外推为 GPU 加速。
6. 模块加速比定义 `median(T_original)/median(T_optimized)`；不同尺寸分别报告，退化也要报告。对 `torch.unique`、histogram 等明确受线程数量影响的操作，不允许 baseline/optimized 使用不同线程数。
7. 单独报告 codec 延迟和包含读写的端到端 wall clock 时间。原始 `compress.py` 使用 CUDA event，而 `decompress.py` 使用 wall clock，二者计时范围不一致；原始解码均值 `[1:-1]` 会让 1–2 个文件得到 NaN，基准程序应修正统计方法并保留原始逐帧数据。
8. 本机若无法加载 CUDA/TorchSparse/原模型，报告中必须将模块 CPU 验证、CUDA 待测、完整 checkpoint 待测分开，不能以算法单测声称完整神经编解码已通过。

整机多倍加速必须来自测得的主要耗时占比。若仅优化一个占比 20% 的模块，即使该模块无限加速，整机上限仍为 1.25×，不能将各模块倍数简单相乘。

## 5. 本次 I/O 实现与测量范围

`src/io.py` 已采用 NumPy 解析 ASCII PLY，依据 header 的 vertex 数和命名 x/y/z 属性确定数据范围与列号。输入 PLY 保持原来的 float64；输入 BIN 保持 float32 XYZI→XYZ。输出通过 `numpy.savetxt` 写 ASCII PLY，先转 float32，以 9 位有效数字保证有限 float32 数值读回精确；文字和文件大小可与原写出不同。写出直接覆盖文件，不再执行 shell 删除后追加。读取多个文件使用最多 8 个默认工作线程，保留输入顺序。

单测 `tests/test_io.py` 共 23 项，覆盖 4 份真实样例的原/新读取精确一致、额外属性和 XYZ 列重排、face 排除、vertex 前置其他元素、空/单点、损坏头和截断、BIN float32、包含负零和非规格数的随机 float32 位模式往返、重复写入、空格路径、并发顺序和异常传播。

`benchmarks/bench_io.py` 可测每个真实样例的 PLY 读取/写出、四样例顺序批量读取和自适应线程批量读取。原函数直接由冻结源码 AST 编译执行；原写出仅用于创建无用对象的 open3d 部分以空实现代替，这排除了原实现的额外开销，因而不会夸大新版本收益。原始 `Pool(64)` 缺少 Windows 主入口保护，无法作为正常脚本基准，因此批量读取行的基线是原单文件函数的顺序调用，报告中须注明。

测量为相同电脑、预热缓存下的 wall clock 时间，包含打开/关闭与读写格式转换，不执行 fsync，不代表冷盘速度或 GPU/完整神经编解码速度。默认预热 2 次、正式交替计时 7 次，保存中位数、p10/p90 和原始各轮时间到 `reports/io_benchmark.json`，平面汇总到同名 CSV。运行命令：

```powershell
& C:\Users\LinkZelda\miniconda3\python.exe benchmarks\bench_io.py --repeats 7
& C:\Users\LinkZelda\miniconda3\python.exe -m pytest tests\test_io.py -q --basetemp D:\Embedded-System\ELiCv1\reports\pytest_io_temp
```

## 6. 本机 I/O 实测

2026-10-15，在 Windows 11、Python 3.13.12、NumPy 2.4.2 环境下，对各项预热 2 次，交替测量原/新实现各 7 次。测量期间没有同时运行其他模块的重型基准。以下为中位数毫秒；完整各轮数据及 p10/p90 已保存到 `reports/io_benchmark.json`，对应 CSV 为 `reports/io_benchmark.csv`。

| 模块 | 输入 | 点数 | 原实现，ms | 新实现，ms | 加速比 |
| --- | --- | ---: | ---: | ---: | ---: |
| PLY 读取 | Ford sample_0001 | 80,332 | 157.899 | 38.809 | 4.069× |
| PLY 读取 | Ford sample_0002 | 80,278 | 175.572 | 47.234 | 3.717× |
| PLY 读取 | SemanticKITTI sample_0001 | 126,012 | 334.948 | 67.907 | 4.932× |
| PLY 读取 | SemanticKITTI sample_0002 | 125,756 | 368.351 | 77.658 | 4.743× |
| PLY 写出 | Ford sample_0001 | 80,332 | 198.480 | 131.887 | 1.505× |
| PLY 写出 | Ford sample_0002 | 80,278 | 200.858 | 132.520 | 1.516× |
| PLY 写出 | SemanticKITTI sample_0001 | 126,012 | 310.655 | 206.099 | 1.507× |
| PLY 写出 | SemanticKITTI sample_0002 | 125,756 | 346.674 | 232.312 | 1.492× |
| 四文件顺序读取 | 全部实际样例 | 412,378 | 1,082.871 | 238.089 | 4.548× |
| 四文件自动线程读取 | 全部实际样例 | 412,378 | 1,036.688 | 230.501 | 4.498× |

两条批量行的原实现均为原单文件读取函数的顺序调用。它们独立采样，基线中位数存在波动，因此不能由 4.548× 与 4.498× 推断线程更慢。新实现实际耗时为顺序 238.089 ms、默认 4 线程 230.501 ms；线程收益较小，保留 `workers=1` 显式选择顺序的选项。此处没有给出相对于原 `Pool(64)` 的倍数，也没有测量冷盘、BIN 读取或 GPU 编解码加速。

I/O 单测结果为 **23 passed**。这证明所覆盖的输入/输出契约及真实样例读取兼容；完整 checkpoint、神经网络 CDF 与码流交叉解码需要单独的 codec 回归，不能由这些 I/O 结果替代。
