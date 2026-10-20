# 命令行、配置与数据接口

## 入口与兼容要求

主入口为 `python portable_bench/run.py`，Python 最低版本为 3.10。入口采用子命令；无参数或 `--help` 显示帮助。它可以从任意工作目录通过脚本绝对路径启动。所有显式指定的路径都支持空格；命令行中仍须按当前 shell 的规则加引号。

`run` 依赖启用状态下的 Python 断言，禁止 `python -O`、`python -OO` 和启用 `PYTHONOPTIMIZE`；工具会显式拒绝并返回代码 2。`doctor` 不受此限制。基础执行要求 little-endian 主机，big-endian 会被前置检查拒绝。

`--repo-root` 指向 ELiCv1 根目录，其中应有 `src/`、`csrc/`、`benchmarks/reference/`。未指定时，优先使用分发包中与 `portable_bench/` 并列的 `runtime/`，否则使用 `portable_bench/` 的父目录。

## `doctor`

```console
python portable_bench/run.py doctor [--repo-root PATH] [--output JSON]
```

输出解释器、系统、依赖、编译器和 CUDA 可用性等信息。它可在未安装 NumPy/PyTorch 时运行，不安装依赖，不编译扩展。缺少某个可选依赖表示相应套件不可用，并不表示主机不能运行全部 CPU 测试。`--output` 另外保存 JSON，便于测试人员回传环境信息。

## `run`

```text
python portable_bench/run.py run
  [--config FILE] [--repo-root PATH] [--output-dir DIR]
  [--device auto|cpu|cuda|cuda:N|cuda-sim]
  [--suites modules io arithmetic geometry binding pipeline|all]
  [--points INT] [--channels INT] [--posq 4 8 16 32 64]
  [--warmups INT] [--repeats INT] [--seed INT] [--threads INT]
  [--compiler PATH] [--files PATH ...] [--prequantized auto|yes|no]
  [--synthetic|--no-synthetic] [--correctness-only|--no-correctness-only]
  [--timeout SECONDS] [--async-entropy|--no-async-entropy]
```

参数表中，`[]` 表示可选。`--posq` 与 `--suites` 后可写多个值；并非要求把表中的所有量化尺度都传入。

| 参数 | 默认值 | 约定 |
| --- | --- | --- |
| `--config` | 无 | UTF-8 JSON 对象；命令行中的显式参数覆盖配置 |
| `--repo-root` | 自动定位 | 被测实现及冻结基线的根目录 |
| `--output-dir` | 工具目录的 `results/` 下独立目录 | 存放本次结果、日志和工作文件 |
| `--device` | `auto` | 自动选择可用 CUDA，否则 CPU；`cuda:N` 选择设备索引 |
| `--suites` | `modules io arithmetic geometry` | 选择套件；`all` 还包含两个可选集成套件 |
| `--points` | `4096` | 合成工作量/模块输入规模；不改变源点云文件 |
| `--channels` | `32` | 合成特征通道数，正整数 |
| `--posq` | `16` | 量化步长，值限定为 `4/8/16/32/64` |
| `--warmups` | `2` | 不计入统计的预热轮数，非负整数 |
| `--repeats` | `7` | 计入统计的重复次数，正整数 |
| `--seed` | `11` | 合成数据随机种子 |
| `--threads` | `1` | PyTorch CPU 线程数，正整数 |
| `--compiler` | 自动检测 | C++17 编译器可执行文件；不会下载编译器 |
| `--files` | 仓库样例 | 一个或多个点云路径/glob；由 Python 展开并排序 |
| `--prequantized` | `auto` | `yes` 为整数坐标；`no` 为原始物理坐标；`auto` 使用样例/文件配置的约定 |
| `--synthetic` | `false` | 使用由固定种子生成的数据，适合不带样例的分发包 |
| `--correctness-only` | `false` | 执行正确性检查，跳过性能采样 |
| `--timeout` | `600` 秒 | 可选外部集成任务的超时；不是整次运行的统一时限 |
| `--async-entropy` | `false` | 可选完整流水线测试使用异步熵编码路径 |

三个布尔选项都支持对应的 `--no-...` 形式，用于覆盖配置文件中的 `true`。`--timeout` 是外部集成子进程的预算；原生 C++ 编译阶段另有 180 秒限制。

`--suites all` 会要求完整网络所需的额外环境。只想运行默认可移植测试时省略 `--suites` 即可。文件列表省略或为 `null` 时自动收集 `repo_root/data/*Example/*.ply`；显式空数组 `[]` 不读这些样例，几何套件使用合成输入。`synthetic: true` 会忽略文件列表。

## 测试套件与已有接口

| 套件 | 对齐的接口与范围 | 额外条件 |
| --- | --- | --- |
| `modules` | `src/morton.py`、`src/op.py` 等张量工具，与冻结原始实现比较 | NumPy、PyTorch |
| `io` | `src/io.py` 的点云读取/写入与批量接口，原始实现读取逻辑与格式检查 | NumPy、PyTorch；基础路径无需 Open3D |
| `arithmetic` | `csrc/arithmetic_core.h`、原始算术编码器、单次及分块编解码、跨实现码流 | C++17 编译器 |
| `geometry` | 量化坐标、Morton 层级/占用序列与算术流的往返；不运行训练网络 | NumPy、PyTorch、C++17 编译器 |
| `binding` | 实际 PyTorch C++ 扩展公开 API 的编译与输入检查 | C++ 工具链、PyTorch 开发环境、setuptools |
| `pipeline` | 仓库的完整网络编码、解码及跨版本解码脚本 | 真实 CUDA、TorchSparse、Open3D、Pandas、Ninja、模型与样例 |

每个套件的具体用例与断言以 JSON 结果的 `cases` 为准。基线来自 `benchmarks/reference/`，不会从网络临时获取；测试也不会修改基线。

## JSON 配置

配置键采用命令行选项对应的下划线名称，例如 `output_dir`、`correctness_only`。数值和布尔值使用 JSON 原生类型，`suites`、`posq` 和 `files` 使用数组。未知字段会被拒绝，以便及时发现拼写错误。

```json
{
  "device": "cpu",
  "suites": ["io", "arithmetic", "geometry"],
  "points": 8192,
  "posq": [4, 16],
  "warmups": 3,
  "repeats": 15,
  "seed": 11,
  "threads": 1,
  "output_dir": "../results/my-run",
  "files": [
    {
      "path": "../../data/FordExample/sample_0001.ply",
      "prequantized": true,
      "scale": 0.001,
      "offset": 131072
    },
    {
      "path": "../../data/SemanticKITTIExample/sample_0001.ply",
      "prequantized": false,
      "scale": 0.001,
      "offset": 131072
    }
  ]
}
```

相对路径以配置文件所在目录为基准解析；命令行直接传入的相对路径以启动时的工作目录为基准。上面的两个样例路径适用于仓库内的 `portable_bench/configs/` 配置目录；分发包里如果显式写样例路径，需改为 `../../runtime/data/...`。附带配置不写死样例路径，以支持自动定位。

文件条目也可直接写字符串，例如 `"files": ["../clouds/*.ply"]`。对象形式便于对同一批数据分别声明坐标约定。

## 点云输入与精度含义

支持 ASCII PLY 和 `.bin`。PLY 由仓库点云 I/O 实现按 `x/y/z` 属性读取；BIN 使用无文件头、本机字节序的 `float32` XYZI，每个点 16 字节，第四列强度不参与几何比较。支持的常见 Windows/Linux/macOS 主机采用 little-endian；本工具不改写原有码流格式，也不转换跨大小端数据。输入格式错误会产生失败结果，不会静默跳过。

旧 I/O 读取器仅能处理其原本支持的简单输入。对于合法的属性重排、附加元素等旧读取器不支持的输入，工具使用独立格式参照校验当前实现；该用例仅报告正确性，`timing` 为 `null`，不计算与旧读取器的配对加速比。基线可处理的输入继续执行逐值对比与计时。

Ford 样例按预量化整数坐标处理；SemanticKITTI 样例按物理坐标处理。自定义数据建议明确指定 `prequantized`，以免文件名推断与实际坐标约定不同。

几何适配保留读取接口的坐标精度，按生产流程处理：未预量化时先在 NumPy 中计算 `points / scale + offset` 再转为张量；已预量化时仅在最小坐标为负时加 `offset`，否则保留原坐标。`scale` 默认 `0.001`，必须有限且为正；`offset` 默认 `131072`，必须为整数。随后计算 `round(xyz / posq)`，采用 ties-to-even 舍入，并去重。结果必须位于 `[0, 2**21)`；几何套件不接受空点云。精确几何正确性针对这一**预期量化后的唯一点集**，不是对有损量化前的浮点坐标要求逐位相等。点的排列顺序不作为几何错误。

`pipeline` 复用的旧完整流水线验收脚本仅接受默认 Ford/SemanticKITTI 样例与对应权重，不能用 `--files` 测试任意点云，也不能用 `--synthetic`。自定义点云可用默认 I/O 和几何套件；完整流水线的量化/坐标约定固定为原脚本约定。

模块测试采用相应接口的数值比较；算术编码要求符号精确一致，兼容性要求码流及跨版本解码满足各用例的断言。此处“正确率”是检查通过比例，不是神经网络分类准确率、PSNR 或 BD-rate。

## 退出码

| 代码 | 含义 |
| --- | --- |
| `0` | 选择的测试均成功 |
| `1` | 至少一个正确性断言或执行任务失败 |
| `2` | 配置错误、依赖/编译器缺失，或所需硬件能力不可用 |
| `130` | 用户取消 |

CI 应同时保存结果目录和进程退出码。运行失败时保留的日志可以区分错误输入、编译错误与算法结果差异。
