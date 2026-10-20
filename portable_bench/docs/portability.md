# CPU、CUDA 与跨平台约定

## 支持范围

基础测试针对 Windows、Linux 和 macOS 的 Python 3.10+ 环境设计。Python 脚本使用 `pathlib`、参数数组和平台编译器，不依赖固定盘符或仓库所在绝对路径。分发包不携带编译好的 DLL/SO，不携带 Python 虚拟环境，因此在接收者主机上以本机编译器构建算术编码测试库。

本次实际验收在 Windows CPU 上完成，包含无 GPU 时的能力缺失处理和接口模拟。Linux/macOS 的路径、编译参数与安装说明已提供，但本次没有在这些系统上实际运行；当前主机也未执行真实 CUDA/TorchSparse 完整网络验收。

只支持 little-endian 测试主机。原项目点云 BIN 和部分流打包接口使用本机字节序；跨用户分发不代表转换了原有码流的大小端格式。`run` 会拒绝 big-endian 主机，以免生成错误的可移植性结论。

| 运行方式 | 计算设备 | 计时说明 | 可验证范围 |
| --- | --- | --- | --- |
| `--device cpu` | CPU | 主机延迟，无 GPU 时间 | 默认四套件及可选 CPU 绑定验证 |
| `--device auto` | 有可用 CUDA 则 CUDA，否则 CPU | 按实际设备测量已完成操作的主机延迟 | 结果中记录实际选择 |
| `--device cuda` / `cuda:N` | 真实 NVIDIA CUDA 设备 | 同步设备后的主机延迟，非纯 kernel 时间 | 张量模块/几何路径及可选完整网络 |
| `--device cuda-sim` | CPU | 全部性能统计为 `null` | 设备选择接口、数据接口与正确性 |

`cuda-sim` 不需要安装 CUDA，也没有硬件性能模型。模拟模式禁止给出 GPU 时间和加速比，以免把 CPU 耗时误认为显卡性能。CPU 算术编码器及文件 I/O 在真实 CUDA 模式下仍是 CPU/主机工作，不能把整行结果都解释为 GPU kernel 用时。

## 为什么没有通用的 PyTorch CUDA 硬件模拟

PyTorch 的 CUDA 张量路径使用真实 CUDA 运行时及设备，参见 [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html)。本项目的完整网络还依赖 TorchSparse 的 CUDA 扩展。

[Numba CUDA Simulator](https://nvidia.github.io/numba-cuda/user/simulator.html)可在 CPU 上模拟特定 Numba kernel 的执行逻辑；其文档明确列出了模拟限制，包括无法链接 PTX。由此可知，把该模拟器启用到当前 Python 环境不能直接执行本项目已有的 PyTorch/TorchSparse CUDA 路径。这是针对本项目依赖方式的判断，工具没有安装或调用 Numba。

如需 GPU 正确性与性能结果，应将分发包交给真实 GPU 机器，在同一份配置下运行。GPU 云主机、实验室机器或具有 GPU runner 的 CI 均可使用相同入口。Docker 本身也不提供硬件模拟；启用 NVIDIA GPU 的容器需要主机显卡、驱动和相应容器支持，见 [NVIDIA Container Toolkit 安装说明](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。

## 编译器、依赖与环境隔离

测试不会调用 pip、conda 或系统包管理器。`doctor` 输出用于确定主机缺少哪些能力；只安装需要运行的套件对应的依赖。

不要用 Python 的 `-O`、`-OO` 或启用 `PYTHONOPTIMIZE` 执行测试。这会删除 Python 断言，`run` 因此显式返回能力缺失结果和退出码 2；环境诊断 `doctor` 仍可使用。

- 默认 CPU 路径：NumPy、CPU 版 PyTorch、C++17 编译器。
- `binding`：另需 setuptools 和可用的 PyTorch C++ 扩展构建环境。
- `pipeline`：另需真实 CUDA、兼容 TorchSparse、Ninja、Pandas、Open3D、tqdm、模型权重和原始样例。

TorchSparse 与 PyTorch/CUDA 的二进制接口需要匹配，因此未在通用 requirements 中强制安装某个 TorchSparse wheel。PyTorch 官方 [安装页面](https://pytorch.org/get-started/locally/)用于选择当前主机对应的 CPU/CUDA wheel。requirements 的下界描述工具使用的基础 API，不保证任意未来版本和任意 TorchSparse 版本组合已经验证。

Windows 不要求把系统 C++ 编译器安装到固定路径；可在 Visual Studio 开发者命令行中启动，或显式指定 `--compiler`。macOS 的 Metal/MPS 设备不属于本工具 CUDA 模式，不会用于生成 CUDA 测速结果。

## 可比较的性能结果

在同一次运行中比较基线和优化版，使用相同输入、量化尺度、线程数和种子。预热不计入统计；保留重复测量原始值，优先观察中位数。真实 GPU 测试需要等待设备工作完成再记录对应时间，防止把异步提交开销当成计算耗时。

跨用户回传时应同时附上 `result.json` 和 `doctor` 输出；操作系统、CPU/GPU、PyTorch 版本、线程数、驱动与负载不同，绝对时间自然不同。一次运行的性能收益不是在所有硬件上的保证。没有实际测量的字段使用 `null`，能力缺失单独标记。
