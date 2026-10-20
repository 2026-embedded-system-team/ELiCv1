# ELiC 自动化测速与正确性检查

这个目录提供独立的测试入口：同一组数据比较原始实现与当前实现，记录正确率、原始耗时、统计结果和运行环境，并生成 JSON、CSV 与 JUnit XML。源码、样例和脚本可以打成 ZIP 分发。执行测试不需要修改优化报告。

先查看环境，再运行 CPU 测试：

```console
python portable_bench/run.py doctor
python portable_bench/run.py run --device cpu --synthetic
```

`doctor` 和打包脚本仅依赖 Python 标准库；运行默认测试还需要 NumPy、PyTorch 和 C++17 编译器。脚本不会自动安装软件。

## 安装与平台

使用 Python 3.10 或更新版本。建议为工具单独创建虚拟环境。以下命令均在 `ELiCv1` 根目录或解压后的 `elic-portable-bench` 根目录执行。

本次实际验证环境是 Windows CPU；Linux/macOS 提供相应路径与安装说明，尚未在本次主机上验证。真实 CUDA 与完整 TorchSparse 网络需要在 GPU 主机进一步验收。

执行 `run` 时不要使用 Python 的 `-O`、`-OO` 或启用 `PYTHONOPTIMIZE`，以免关闭正确性断言；工具会拒绝该环境并以代码 2 退出。`doctor` 仍可运行。支持的测试主机采用 little-endian 字节序；已有二进制码流保留本机格式，不提供跨大小端转换。

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r portable_bench/requirements.txt
.\.venv\Scripts\python.exe portable_bench/run.py doctor
.\.venv\Scripts\python.exe portable_bench/run.py run --device cpu --synthetic
```

算术编码测试需要 C++17 编译器。Windows 可使用 Visual Studio Build Tools 的 C++ 工具链；也可使用安装并加入 PATH 的兼容 GCC/Clang。工具会检测环境；必要时通过 `--compiler` 指定编译器的绝对路径。

Linux：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r portable_bench/requirements.txt
.venv/bin/python portable_bench/run.py run --device cpu --synthetic
```

系统需提供 GCC 或 Clang。macOS 使用 `xcode-select --install` 安装命令行工具，再使用 `python -m pip install torch` 安装适用于 macOS 的 PyTorch；其余用法相同。macOS 的本工具路径为 CPU；MPS 不属于 CUDA 测试。

真实 GPU 环境应按 [PyTorch 安装选择器](https://pytorch.org/get-started/locally/)安装与平台匹配的 PyTorch；不要先安装上面的 CPU wheel。需要完整网络编解码时，再安装 `requirements-pipeline.txt` 和与 PyTorch/CUDA 匹配的 TorchSparse。基础 CPU 测试无需 TorchSparse 或 Open3D。

## 常用命令

```console
# 四个默认套件：modules、io、arithmetic、geometry。
python portable_bench/run.py run --device cpu

# 固定工作量、预热、重复次数；使用仓库样例点云。
python portable_bench/run.py run --device cpu --points 16384 --warmups 3 --repeats 15

# 自己的 PLY 或 float32 XYZI BIN；路径中的空格需要引号。
python portable_bench/run.py run --device cpu --files "my data/*.ply" --prequantized no

# 只检查正确性，不产生性能结论。
python portable_bench/run.py run --device cpu --synthetic --correctness-only

# 实际 CUDA 张量测试，需要真实可用的 CUDA 设备。
python portable_bench/run.py run --device cuda --suites modules geometry

# 没有 GPU 时检查 CUDA 设备选择接口的 CPU 回退路径。
python portable_bench/run.py run --device cuda-sim --synthetic --correctness-only

# 可选的真实 PyTorch C++ 绑定验证。
python portable_bench/run.py run --device cpu --suites binding

# 可选的完整训练模型编解码与跨版本解码；需真实 CUDA、依赖、样例、权重。
python portable_bench/run.py run --device cuda --suites pipeline --repeats 3

# 用文件保存参数，命令行参数覆盖同名配置项。
python portable_bench/run.py run --config portable_bench/configs/cpu.json
```

没有真实 CUDA 时，`--device cuda` 会写出能力缺失结果并以代码 2 退出。`cuda-sim` 只把计算映射到 CPU，用于测试接口与正确性；没有 GPU 时间，也不能运行 CUDA kernel 或 TorchSparse 完整模型。详见 [CUDA 与移植限制](docs/portability.md)。

## 结果与分发

默认结果写入 `portable_bench/results/` 下的独立运行目录，包含 `result.json`、`results.csv`、`junit.xml`，按需保留日志与集成工作文件；临时本机编译产物会在结束时清理。JSON 保留每轮原始数据与环境信息，CSV 用于比较，JUnit XML 可接入 CI。`--output-dir PATH` 可指定新目录，已有结果不会被覆盖。正确率是已完成的完整用例通过比例，每个用例可以包含多个断言；某个能力缺失不会伪装成测试通过。

创建包含源码和四个点云样例的分发包：

```console
python portable_bench/bundle.py --repo-root . --output portable_bench/dist/elic-portable-bench.zip
python portable_bench/bundle.py --repo-root . --output portable_bench/dist/elic-full.zip --include-models
python portable_bench/bundle.py --repo-root . --output portable_bench/dist/elic-small.zip --no-samples
python portable_bench/bundle.py --verify portable_bench/dist/elic-portable-bench.zip
```

解压后仍使用 `python portable_bench/run.py ...`；工具会自动找到旁边的 `runtime/`。包中保留原项目许可证，并提供逐文件 SHA-256 清单。默认包不带模型权重；完整网络测试可用 `--include-models` 打包权重。详见 [分发说明](docs/distribution.md)。

完整接口说明：

- [命令行、配置和数据约定](docs/interfaces.md)
- [结果格式、统计口径与扩展接口](docs/results-and-api.md)
- [CPU、CUDA、模拟接口及移植限制](docs/portability.md)
- [打包、解压、校验和 CI 接入](docs/distribution.md)
