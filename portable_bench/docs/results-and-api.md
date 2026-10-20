# 结果、计时与 Python 扩展接口

## 文件与版本

每次 `run` 都将完整结果写入 `result.json`，扁平概览写入 `results.csv`，CI 断言结果写入 `junit.xml`。结果格式版本是 `schema_version: "1.0"`；工具自身版本单独写在 `tool.version`。机器读取端应检查格式版本，并允许以后增加字段。可选的形式化结构见 [result.schema.json](result.schema.json)，无需安装 JSON Schema 库才能执行测试。

JSON 顶层字段：

| 字段 | 含义 |
| --- | --- |
| `schema_version` | 结果接口版本字符串 |
| `tool` | 工具名称和代码版本 |
| `created_utc` | 本次运行创建时间，带时区的 ISO 8601 UTC 字符串 |
| `config` | 规范化参数，含解析后的绝对路径和文件坐标约定 |
| `environment` | 解释器、系统、依赖版本、编译器路径和 CUDA 设备信息 |
| `backend` | 请求设备、实际设备、是否接口模拟等；前置检查失败时可为 `null` |
| `provenance` | 冻结基线校验、当前源文件 SHA-256、测试工具源码 SHA-256、输入文件 SHA-256 |
| `native_build` | 运行需要编译时的编译器、参数、源文件哈希、构建耗时等 |
| `cases` | 各用例的正确性、计时、范围与失败原因 |
| `summary` | 状态、退出码、用例计数、正确率和覆盖是否完整 |
| `elapsed_seconds` | 整次任务的实际持续时间，包含构建和正确性检查 |

`provenance` 用于判断不同用户测试的是否为同一份实现和输入。其中 `candidate_source_sha256` 保存被测源码的逐文件哈希，`tool_source_sha256` 保存测试工具源码的逐文件哈希，`input_sha256` 保存文件输入哈希。基线清单校验失败时不会继续给出可信的优化对比。运行期间被测源码或测试工具源码变更也会导致结果失败，避免把不同代码混在一次测量中。

## 单个用例

以下为结构示例；数值只是格式演示，并非实际测速结果：

```json
{
  "case_id": "example.operation",
  "suite": "modules",
  "status": "passed",
  "execution_kind": "hardware",
  "cuda_execution_validated": false,
  "correctness": {
    "check_passed": true,
    "metrics": {"exact": true}
  },
  "details": {"uses_cuda": false},
  "timing": {
    "unit": "ms",
    "clock": "perf_counter_ns",
    "aggregation": "median",
    "order": "alternating baseline/candidate",
    "warmups": 2,
    "repeats": 3,
    "baseline": {
      "median_ms": 2.0,
      "p10_ms": 1.92,
      "p90_ms": 2.08,
      "min_ms": 1.9,
      "max_ms": 2.1,
      "samples_ms": [1.9, 2.0, 2.1]
    },
    "candidate": {
      "median_ms": 1.0,
      "p10_ms": 0.92,
      "p90_ms": 1.08,
      "min_ms": 0.9,
      "max_ms": 1.1,
      "samples_ms": [0.9, 1.0, 1.1]
    },
    "cuda_synchronized": false,
    "speedup": 2.0,
    "timing_scope": "operation call including returned-output allocation"
  }
}
```

| 状态 | 含义 | JUnit 表达 |
| --- | --- | --- |
| `passed` | 正确性断言通过 | 普通 testcase |
| `failed` | 结果或接口断言不成立 | failure |
| `error` | 执行、构建或数据准备过程发生错误 | error |
| `unavailable` | 缺少指定能力，例如没有真实 CUDA | skipped |
| `skipped` | 选择的用例被跳过 | skipped |

`reason` 为可读说明，异常用例可带 `traceback`。`correctness.check_passed` 为 `true/false/null`；`null` 表示该检查未完成，不等于正确。`summary.correctness.pass_rate` 是通过的完整正确性用例数除以已完成检查用例数；没有完成检查时为 `null`。查看此比例时必须同时检查 `summary.coverage_complete` 和状态计数。个别套件还会在 `metrics` 中记录 `symbol_accuracy`、`point_accuracy`、精确点数或四种跨版本编解码组合是否通过。

`cuda_execution_validated: true` 仅说明对应通过用例确实运行了指定 CUDA 张量/模型路径。它不是 PyTorch C++ 绑定或全部网络路径已经验证的总标记。接口模拟中为 `false`。

## 计时口径

默认套件先执行正确性检查，再预热、采样。基线与当前版交替先后执行，减小执行顺序偏差。基线和当前版分别统计中位数、P10/P90、最小值、最大值和原始轮次，P10/P90 使用排序后的线性插值。

`speedup = baseline.median_ms / candidate.median_ms`，大于 1 表示当前版更快。每行代表自己的操作范围，不能把所有模块行相加当成完整模型总耗时。

使用主机 `perf_counter_ns`，单位转成毫秒；CUDA 模式在操作前后同步实际设备。因此默认套件的时间包含调用、分配、必要传输和同步，属于已完成操作的主机延迟，不是 CUDA Event 纯 kernel 时间。CPU 算术编码始终在 CPU 执行，其范围另在 `details` 与 `timing_scope` 中记录。文件读写可能命中操作系统缓存，没有强制清空磁盘缓存或 fsync，不能用它推导冷盘吞吐量。

原始算术编码器通过移除 Torch 适配层接入 NumPy/ctypes；码流核心算法保留。基线 Morton 查表在实际设备上预先创建；I/O 原始函数通过 AST 提取，写入中未使用的 Open3D 对象转换使用占位适配。相关适配均以可移植性为目的，测试结果应结合每行的 `details.baseline`/`scope` 阅读。

合法自定义输入若超出旧 I/O 读取器能力，则使用独立格式参照校验当前读取器，保留正确性结果，`timing` 为 `null`。这些用例没有有效的基线性能数据，不会生成配对时间或加速比；不要将它们解释成旧版本的算法回归。

纯断言用例、`--correctness-only` 和全部 `cuda-sim` 用例的 `timing` 为 `null`。`case_wall_seconds` 和顶层 `elapsed_seconds` 是调度/诊断用的持续时间，不是 GPU 测速。原生编译耗时保存在 `native_build`，不计入基线与当前版的操作时间，构建目录中的临时本机库会在结束时清理。

完整 `pipeline` 套件复用已有接入脚本，统计保存在 `details.integration_result.rows`。其原始脚本涉及基线 CUDA Event 与当前版同步主机计时，整个进程另记墙钟耗时；这些统计与默认套件的统一操作计时不是同一口径。完整流水线的 `--correctness-only` 使用一次验收重复并移除返回结果中的性能统计，记录 `performance_measurements_suppressed: true`。`binding` 只验证生产 Torch 绑定正确性。两者顶层 `timing` 均为 `null`，具体能力和范围见各自 `details`。

算术合成分布为控制内存将符号数限制到 65536；`details.requested_symbols` 和 `details.symbols` 分别记录请求与实际值。真实点云几何测试读取所选文件的全部点，`--points` 不会截断这些点云。

## CSV 与 CI

CSV 使用 UTF-8 BOM，列为：

```text
case_id,suite,status,execution_kind,cuda_execution_validated,check_passed,baseline_median_ms,candidate_median_ms,speedup,reason
```

没有性能测量的列留空；完整指标、每轮采样和集成子结果以 JSON 为准。JUnit 把能力缺失写成 skipped；命令行仍会以代码 2 表示覆盖不完整，因此 CI 应检查进程退出码。

## 公共 Python API

在仓库根或解压包根运行：

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("portable_bench").resolve()))
from elic_bench.api import discover_environment, run_benchmarks

environment = discover_environment()  # 仅诊断，不安装软件
report = run_benchmarks({
    "device": "cpu",
    "synthetic": True,
    "suites": ["modules", "io", "arithmetic", "geometry"],
    "repeats": 3,
    "output_dir": "api-result",
})
raise SystemExit(report["summary"]["exit_code"])
```

`discover_environment(repo_root=None) -> dict` 接受可选源码目录。`run_benchmarks(config=None) -> dict` 接受与 JSON 配置相同的字段，完成后返回完整报告并写出三个结果文件；它自身不退出 Python 进程。直接调用 API 时，相对路径以当前工作目录为基准。配置无效会抛出 `elic_bench.config.ConfigError`，运行期故障记录到返回报告中。已存在 `result.json` 的结果目录会被拒绝覆盖，请为独立运行使用不同目录。

## 自定义被测实现适配

扩展入口为 `elic_bench.context.Context`，无需改动生产源码。调用者提供无参数的基线/当前版函数，并在计时之前构造好相同输入：

```python
import torch
from elic_bench.backend import Backend
from elic_bench.config import normalize_config
from elic_bench.context import Context
from elic_bench.results import summarize, write_outputs

config = normalize_config({"device": "cpu", "synthetic": True,
                           "output_dir": "custom-results", "repeats": 5})
ctx = Context(config, Backend(config["device"], torch))
x = torch.arange(1024, dtype=torch.float32)

def check(a, b):
    assert torch.equal(a, b), "custom result differs"
    return {"exact": True}

ctx.record_case("custom.sum", "custom", lambda: sum(x), lambda: x.sum(),
                check=check, details={"uses_cuda": False,
                                      "timing_scope": "sum of prepared tensor"})
report = {"schema_version": "1.0", "cases": ctx.records,
          "summary": summarize(ctx.records)}
write_outputs(report, config["output_dir"])
```

上面的自定义简报适用于 `write_outputs`；如需满足完整 JSON Schema 的顶层环境/来源字段，应基于 `run_benchmarks` 的报告结构补齐。当前没有命令行动态插件加载参数，示例是 Python 适配接口。要将自定义套件加入主入口，需在配置的套件名单与 API 分发中显式注册。

`record_case(case_id, suite, before, after, check=None, details=None) -> dict` 会立即把唯一 ID 的记录加入 `ctx.records`。不要把返回值再次追加。`check(before_value, after_value)` 可用断言报告差异，返回字典会保存到 `correctness.metrics`；返回字面值 `False` 表示失败。不提供 `check` 时要求结构、数组/张量形状、dtype 和数值完全一致。断言失败是 `failed`，其他异常是 `error`。

`details.uses_cuda` 必须如实声明；要只做断言，设置 `details.timing_scope = "correctness_only"`。`ctx.add_result(case_id, suite, status, reason=None, **extra)` 用于表达能力缺失等非计时用例。函数应能够重复调用，输入构造应放在函数外，输出分配是否纳入时间应写明；有状态编码器在每轮开始前须恢复到相同状态。

## NumPy 算术编解码适配器

`elic_bench.arithmetic.build_coders(repo_root, build_dir, compiler=None)` 是上下文管理器，返回 `(reference, candidate, build_metadata)`。两份对象具有与生产 Torch 算术绑定对应的调用形式，数据类型改为严格的 NumPy 数组；离开上下文后原生句柄不可再用。

| 接口 | 输入/输出 |
| --- | --- |
| `encode_cdf_16(cdf, symbols)` | 返回 `bytes` |
| `decode_cdf_16(cdf, bitstream)` | 返回 shape `(N,)` 的 `np.int16` |
| `ArithmeticEncoder16(validate=True)` | 可作为上下文管理器，结束时关闭句柄 |
| `encoder.encode_chunk(cdf, symbols)` | 追加同一条编码流，允许空块 |
| `encoder.finish(reset_after=True, clear_output_after=True)` | 返回当前完成的流，默认重置状态和输出 |
| `encoder.reset(reset_output=True)` | 重置编码状态，可选择保留输出 |
| `encoder.get_buffer_bytes()` / `buffered_bitcount()` | 读取缓冲状态 |
| `ArithmeticDecoder16(validate=True)` | 构造解码器，可作为上下文管理器 |
| `decoder.reset(bitstream)` | `bytes` 开始一条独立流 |
| `decoder.decode_chunk(cdf)` | 返回该块的符号 |
| `decoder.bytes_consumed()` / `bytes_remaining()` | 读取流消耗状态 |
| `close()` | 显式关闭编码/解码句柄 |

CDF 必须为 native-endian、C 连续的 `np.int16`，shape 为 `(N,17)`；每行表示 16 个符号的 uint16 累积边界，使用 `unsigned_cdf.view(np.int16)` 保留位模式。首项是 0，前 16 项严格递增，末项允许用 0 表示 65536。符号必须为 C 连续 `np.int16`，shape `(N,)`，范围 `0..15`。不进行隐式 dtype 转换、连续化或 GPU 到 CPU 拷贝。`validate=False` 只关闭 CDF 值检查，不关闭形状/dtype/符号范围检查；只能用于已验证的输入。

`compiler` 或环境变量 `CXX` 可指定编译器；否则按可用性检测 `g++`、`clang++`、`cl`。编译器缺失抛出 `NativeUnavailable`，编译或适配错误抛出 `NativeBuildError`；不会替换成假实现继续报成功。
