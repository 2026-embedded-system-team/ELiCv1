# 分发、校验与 CI

## 创建 ZIP

打包工具只依赖 Python 标准库，无需先安装 NumPy、PyTorch 或 CUDA。

```console
python portable_bench/bundle.py --repo-root . --output portable_bench/dist/elic-portable-bench.zip
python portable_bench/bundle.py --repo-root . --output portable_bench/dist/elic-full.zip --include-models
python portable_bench/bundle.py --repo-root . --output portable_bench/dist/elic-small.zip --no-samples
python portable_bench/bundle.py --verify portable_bench/dist/elic-portable-bench.zip
```

`--repo-root` 可指向另一份具有相同源码布局的 ELiCv1；工具源码来自正在执行的 `bundle.py` 所在目录。`--output` 必须是 `.zip`。创建过程写入临时文件、校验、最后原子替换目的 ZIP；目的文件已存在时覆盖。不会向包中递归收集既有 ZIP。

默认收录四个点云 PLY；`--no-samples` 可以减小分发体积，接收者用 `--synthetic` 或提供自己的 `--files` 测试。`--include-models` 额外收录 `model/` 下的 `.pt/.pth/.npy/.json` 文件，包括完整网络测试所需的权重和 BoE 中心；默认包不带权重。

## 布局与收录规则

```text
elic-portable-bench/
  manifest.json
  portable_bench/
    run.py
    bundle.py
    elic_bench/
    configs/
    docs/
    tests/
    requirements*.txt
  runtime/
    LICENSE
    LICENSE.md
    src/
    csrc/
    benchmarks/
      reference/
      ...所需测试脚本与 C++ harness
    tests/
    model.py
    compress.py
    decompress.py
    data/                    # 除非指定 --no-samples
    model/                   # 仅 --include-models
```

使用明确的源码白名单：工具入口、Python 包、文档、配置和测试；当前源码 `src/*.py`、`csrc` 源文件、所需基准 harness、冻结基线及已有原始回归测试。样例路径限定为 Ford 和 SemanticKITTI 各两个 PLY。

不会打包 `.git`、本地虚拟环境、报告、已有测速结果、临时目录、DLL/SO/PYD/EXE 或缓存。保留 `runtime/LICENSE` 和 `runtime/LICENSE.md` 原文。源码不允许通过符号链接逃逸到所选根目录外。运行结果产生在工具的 `results/` 下，重新打包时不会收录。

## 完整性清单

`manifest.json` 的结构如下，所有路径相对于 `elic-portable-bench/`：

```json
{
  "schema_version": 1,
  "created_utc": "2026-10-20T00:00:00+00:00",
  "package": "elic-portable-bench",
  "samples_included": true,
  "models_included": false,
  "hash_algorithm": "sha256",
  "manifest_excludes_itself": true,
  "files": [
    {
      "path": "portable_bench/run.py",
      "bytes": 123,
      "sha256": "文件内容的 SHA-256 十六进制字符串",
      "category": "tool"
    }
  ]
}
```

`files` 覆盖除清单自身之外的全部包内文件。分类为 `tool/runtime/baseline/license/sample/model`。`--verify` 不解压 ZIP，检查路径安全、重复条目、清单与 ZIP 条目完全一致、文件长度与 SHA-256；成功返回代码 0，错误返回代码 2。哈希用于检测损坏和内容变化，并非数字签名。

## 接收者运行

解压到任意可写目录，在 `elic-portable-bench` 目录中执行：

```console
python portable_bench/run.py doctor --output doctor.json
python portable_bench/run.py run --device cpu --synthetic --output-dir results/cpu
```

先按 README 安装相应平台依赖。脚本会自动将 `runtime/` 作为源码根目录。也可指向接收者自己的源码：

```console
python portable_bench/run.py run --repo-root "PATH/TO/ELiCv1" --device cpu --synthetic
```

重新移动整个解压目录后不需要改配置。若自行配置了外部绝对路径，则相应外部文件仍需存在。无需恢复发送者的盘符、编译产物、虚拟环境或输出目录。

## CI 接入

普通 CPU runner 可以先运行环境探测，再执行固定参数。真实 GPU runner 另用 `--device cuda`，没有 GPU 时会明确退出而不是自动冒充 CUDA：

```sh
python portable_bench/run.py doctor --output ci-doctor.json
python portable_bench/run.py run --device cpu --synthetic --warmups 1 --repeats 3 --output-dir ci-results
```

收集 `ci-results/result.json`、`ci-results/results.csv`、`ci-results/junit.xml`、日志与 `ci-doctor.json`。把 JUnit XML 作为测试结果上传，JSON/CSV 作为构建产物保存。性能回归阈值建议由同一型号且负载受控的 runner 设定；本工具不对不同用户的主机设置统一绝对耗时门槛。

打包脚本的标准库测试可独立运行：

```console
python -m unittest discover -s portable_bench/tests -p test_bundle.py -v
```
