"""Consolidate measured JSON results without rerunning performance tests."""
import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEMORY_IO_DIR = 'benchmarks/results/memory_io_20261020'
REPORT_DATE = '2026-10-20'


def read(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spread(values):
    values = list(values)
    lo, hi = min(values), max(values)
    return f'{lo:.2f}×' if len(values) == 1 else f'{lo:.2f}–{hi:.2f}×'


def memory_io_summary(data):
    """Keep the incremental comparison separate from the original baseline."""
    samples = [row for row in data['rows'] if row['case'].startswith('histogram_')]
    assert len(samples) == len(data['geometry_histogram_inputs']) == 4
    before = sum(row['before_ms'] for row in samples)
    after = sum(row['after_ms'] for row in samples)
    assert before == data['geometry_aggregate']['before_sum_medians_ms']
    assert after == data['geometry_aggregate']['after_sum_medians_ms']
    assert digest(ROOT/'csrc/arithmetic_core.h') == data['optimized_sha256']
    return dict(workflow='bitreader_batch_load', date=REPORT_DATE, frames=len(samples),
                baseline_kind='previous optimized implementation immediately before the memory-I/O change',
                baseline_core_sha256=data['baseline_sha256'], optimized_core_sha256=data['optimized_sha256'],
                baseline_total_ms=before, optimized_total_ms=after,
                baseline_mean_ms=before/len(samples), optimized_mean_ms=after/len(samples),
                speedup=before/after, time_reduction_percent=(1-after/before)*100,
                method='Sum per-cloud decode medians, then baseline_sum / optimized_sum',
                source='reports/memory_io_benchmark.json',
                scope='CPU integer decode; posQ=16; actual occupancy symbols and per-layer empirical CDFs',
                included_in_historical_cpu_cost_budget=False)


def memory_io_section(data, experiment, aggregate):
    samples = [row for row in data['rows'] if row['case'].startswith('histogram_')]
    labels = {'histogram_Ford_sample_0001': 'Ford sample_0001',
              'histogram_Ford_sample_0002': 'Ford sample_0002',
              'histogram_SemanticKITTI_sample_0001': 'SemanticKITTI sample_0001',
              'histogram_SemanticKITTI_sample_0002': 'SemanticKITTI sample_0002'}
    table = '\n'.join(f'| {labels[row["case"]]} | {row["before_ms"]:.4f} | {row["after_ms"]:.4f} | '
                      f'{row["speedup"]:.3f}× |' for row in samples)
    uniform = next(row for row in data['rows'] if row['case'] == 'uniform')
    concentrated = next(row for row in data['rows'] if row['case'] == 'concentrated')
    trial = experiment['rejected_tensor_trial']
    return f'''## 2026-10-20 访存 IO 增量优化

本轮将 `csrc/arithmetic_core.h` 的 `BitReader` 从逐字节装载改为一次 `memcpy` 读取相邻 **4 字节**，并用 **64 位缓存**按需取位，减少取数和分支开销。装载允许非对齐地址；不足 4 字节的尾部单独处理，保留字节序、流末零填充、reset 和逻辑已消耗/剩余字节数。

**基线是本次修改前、已经完成上一轮优化的版本**，对应核心 SHA-256 `{data['baseline_sha256']}`；本轮核心为 `{data['optimized_sha256']}`。下表为 CPU 单线程整数算术解码，四份完整点云的真实占用序列、逐层经验 CDF、posQ=16。预热 3 次后交替运行 15 轮，取各帧中位数；计时包含 decoder reset、码流复制和完整 decode_chunk，双方均预分配输出，编译、输入/CDF 生成和正确性检查不计时。

| 点云 | 优化前 ms | 优化后 ms | 增量加速 |
| --- | ---: | ---: | ---: |
{table}
| 四帧中位数合计 | **{aggregate['baseline_total_ms']:.4f}** | **{aggregate['optimized_total_ms']:.4f}** | **{aggregate['speedup']:.3f}×** |

四帧耗时合计下降 **{aggregate['time_reduction_percent']:.2f}%**，折合平均 **{aggregate['baseline_mean_ms']:.4f} → {aggregate['optimized_mean_ms']:.4f} ms/帧**。合计采用各帧中位数求和后相除。均匀合成分布为 **{uniform['speedup']:.4f}×**，约慢 **{(1/uniform['speedup']-1)*100:.2f}%**；极端集中分布为 **{concentrated['speedup']:.4f}×**，未见明显收益，完整分布结果保留在原始记录中。

该结果仅覆盖上述 CPU 整数解码工作负载；神经预测、实际神经 CDF、Torch 绑定和完整 GPU 流程仍待验证。10 月 15 日的原始版本对比和 CPU 成本合计沿用当时测量，本轮结果单列展示，不与此前不同基线、不同轮次的倍数相乘，也不替换历史算术编码＋解码合计。

Morton 的 XYZ 批次处理也做了尝试，真实点云完整层级构建中位数合计 **{trial['aggregate_before_ms']:.4f} → {trial['aggregate_after_ms']:.4f} ms**（**{trial['aggregate_speedup']:.4f}×**），没有稳定提速；另外尝试的 CDF 缓存分块在独立 CDF 基准上也未显示稳定收益。这两项已撤回，生产 `src/morton.py` 和 `src/op.py` 保持本轮修改前版本；[试验记录](../{MEMORY_IO_DIR}/trials/tensor_batch/tensor_benchmark.json)及候选源码已归档。

验证：完整回归 **87 项通过、49 个子测试通过、3 项 CUDA 跳过**；447 项原版整数码流兼容案例通过。新增 **{data['reader_checks']:,} 次**直接取位/位置检查、{data['decoder_cases']} 组解码案例及 {data['geometry_correctness_cases']} 份真实占用流检查，覆盖 0–32 位取数、非整块尾部、空流、截断后的零填充、分块/reset 和逻辑字节计数，均通过。

环境沿用 Windows 11 / Intel Family 6 Model 170，g++ 15.2.0，`-O3 -std=c++17 -DNDEBUG`。完整原始采样、输入与源码哈希、编译参数见 [JSON](memory_io_benchmark.json)，各分布及合计见 [CSV](memory_io_benchmark.csv)，实现细节见 [算术核说明](arithmetic_notes.md)。复现时将新结果写到独立文件：

```powershell
$elicPython = 'C:\\Users\\LinkZelda\\miniconda3\\python.exe'
& $elicPython benchmarks/bench_bitreader_batch.py --before benchmarks/results/memory_io_20261020/before/csrc/arithmetic_core.h --benchmark --geometry-histograms --output benchmarks/results/memory_io_20261020/bitreader_retest.json
```

'''


def overall_summary(morton, boe, io, arithmetic, pipeline):
    """Aggregate existing component medians; do not invent pipeline timings."""
    count = len(morton['samples'])
    frames = {s['file'].replace('\\', '/') for s in morton['samples']}
    assert count > 0 and len(frames) == count
    assert {s['file'].replace('\\', '/') for s in boe['samples']} == frames
    io_rows = [r for r in io['results'] if r['module'] in ('PLY read', 'PLY write')]
    assert len(io_rows) == count*2
    for frame in frames:
        assert sorted(r['module'] for r in io_rows if r['sample'].replace('\\', '/') == frame) == ['PLY read', 'PLY write']
    entropy_rows = [r for r in arithmetic['benchmarks'] if r['distribution'].startswith('histogram_')]
    assert len(entropy_rows) == count

    def row(key, label, old, new, source, scope):
        assert old > 0 and new > 0
        return dict(workflow=key, label=label, frames=count,
                    baseline_total_ms=old, optimized_total_ms=new,
                    baseline_mean_ms=old/count, optimized_mean_ms=new/count,
                    speedup=old/new, time_reduction_percent=(1-new/old)*100,
                    source=source, scope=scope)

    flows = []
    for key, label, data, stage, source, scope in [
        ('hierarchy', '完整几何层级构建', morton, 'build_hierarchy', 'benchmarks/results_morton.json',
         'CPU, posQ=16; includes Morton sorting and all downsampling levels'),
        ('boe_offline', 'BoE 离线完整直方图收集', boe, 'boe_offline_collect', 'reports/boe_cpu.json',
         'CPU, posQ=4; includes its own hierarchy; excludes KMeans/TSNE and data loading')]:
        flows.append(row(key, label, sum(s['stages'][stage]['baseline_ms'] for s in data['samples']),
                         sum(s['stages'][stage]['optimized_ms'] for s in data['samples']), source, scope))
    flows.append(row('io_read_write', 'PLY 读取＋写出',
                     sum(r['baseline_ms'] for r in io_rows), sum(r['optimized_ms'] for r in io_rows),
                     'reports/io_benchmark.json', 'Sum of single-file read and write medians; excludes batch-reader rows'))
    flows.append(row('arithmetic_roundtrip', '整数算术编码＋解码',
                     sum((r['baseline_encode_seconds']+r['baseline_decode_seconds'])*1000 for r in entropy_rows),
                     sum((r['optimized_encode_seconds']+r['optimized_decode_seconds'])*1000 for r in entropy_rows),
                     'reports/arithmetic_benchmark.json', 'Actual occupancy symbols with empirical CDF; excludes neural prediction, CDF generation and Torch adapters'))
    budget_flows = [r for r in flows if r['workflow'] != 'boe_offline']
    budget = row('measured_cpu_cost_budget', '已测 CPU 阶段成本合计（分析值）',
                 sum(r['baseline_total_ms'] for r in budget_flows),
                 sum(r['optimized_total_ms'] for r in budget_flows),
                 [r['source'] for r in budget_flows],
                 'Analytical sum of separate component medians; NOT a timed end-to-end neural codec run')
    return dict(method='Sum per-frame component medians, then baseline_sum / optimized_sum; not average speedups',
                frame_count=count, total_input_points=sum(s['input_points'] for s in morton['samples']),
                workflows=flows, measured_cpu_cost_budget=budget,
                full_neural_pipeline_status=pipeline['status'],
                exclusions=['Neural sparse convolutions and prediction', 'Full decoder upsampling across all levels',
                            'Actual model CDF generation and normalization workload', 'CPU/GPU transfers and synchronization',
                            'Model/JIT startup, total peak memory and full neural compression rate'])


def main():
    morton = read('benchmarks/results_morton.json')
    boe = read('reports/boe_cpu.json')
    io = read('reports/io_benchmark.json')
    modules = read('reports/modules_cpu.json')
    arithmetic = read('reports/arithmetic_benchmark.json')
    pipeline = read('reports/pipeline_validation.json')
    binding = read('reports/arithmetic_torch_binding.json')
    memory_io = read(f'{MEMORY_IO_DIR}/bitreader_final.json')
    experiment = read(f'{MEMORY_IO_DIR}/experiment_summary.json')
    incremental = memory_io_summary(memory_io)
    (ROOT/'reports/memory_io_benchmark.json').write_bytes((ROOT/MEMORY_IO_DIR/'bitreader_final.json').read_bytes())
    with (ROOT/'reports/memory_io_benchmark.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['scope', 'module', 'sample', 'baseline_kind',
                                                   'baseline_ms', 'optimized_ms', 'speedup'])
        writer.writeheader()
        for value in memory_io['rows']:
            writer.writerow(dict(scope='memory_io_incremental_cpu', module='bitreader_decode', sample=value['case'],
                                 baseline_kind=incremental['baseline_kind'], baseline_ms=value['before_ms'],
                                 optimized_ms=value['after_ms'], speedup=value['speedup']))
        writer.writerow(dict(scope='memory_io_incremental_cpu', module='bitreader_decode', sample='four_real_clouds_sum_medians',
                             baseline_kind=incremental['baseline_kind'], baseline_ms=incremental['baseline_total_ms'],
                             optimized_ms=incremental['optimized_total_ms'], speedup=incremental['speedup']))
    overall = overall_summary(morton, boe, io, arithmetic, pipeline)
    overall['historical_measurement_date'] = '2026-10-15'
    overall['incremental_experiments'] = [incremental]
    (ROOT/'reports/overall_summary.json').write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding='utf-8')
    overall_fields = ['workflow', 'label', 'frames', 'baseline_total_ms', 'optimized_total_ms',
                      'baseline_mean_ms', 'optimized_mean_ms', 'speedup', 'time_reduction_percent']
    with (ROOT/'reports/overall_summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=overall_fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(overall['workflows']+[overall['measured_cpu_cost_budget']])
    overall_table = '\n'.join(f'| {r["label"]} | {r["baseline_mean_ms"]:.3f} | {r["optimized_mean_ms"]:.3f} | '
                              f'{r["speedup"]:.3f}× | {r["time_reduction_percent"]:.2f}% |' for r in overall['workflows'])
    budget = overall['measured_cpu_cost_budget']
    suites = ET.parse(ROOT/'reports/tests.xml').getroot().findall('testsuite')
    assert all(int(s.get('failures')) == int(s.get('errors')) == 0 for s in suites)
    log_path = ROOT/'reports/test_output.txt'
    log = log_path.read_text(encoding='utf-16' if log_path.read_bytes().startswith(b'\xff\xfe') else 'utf-8')
    match = re.search(r'(\d+) passed, (\d+) skipped, (\d+) subtests passed', log)
    if match is None:
        raise RuntimeError('Cannot read final pytest summary')
    passed, skipped, subtests = map(int, match.groups())
    rows = []
    for sample in morton['samples']:
        for stage, values in sample['stages'].items():
            rows.append(dict(scope='real_geometry_cpu', module=stage, sample=sample['file'],
                             baseline_ms=values['baseline_ms'], optimized_ms=values['optimized_ms'], speedup=values['speedup']))
    for sample in boe['samples']:
        for stage, values in sample['stages'].items():
            rows.append(dict(scope='offline_boe_cpu', module=stage, sample=sample['file'],
                             baseline_ms=values['baseline_ms'], optimized_ms=values['optimized_ms'], speedup=values['speedup']))
    for values in io['results']:
        rows.append(dict(scope='io_cpu', **{key: values[key] for key in ('module', 'sample', 'baseline_ms', 'optimized_ms', 'speedup')}))
    for values in modules['rows']:
        rows.append(dict(scope='synthetic_100k_cpu', sample=str(modules['points']),
                         **{key: values[key] for key in ('module', 'baseline_ms', 'optimized_ms', 'speedup')}))
    for values in arithmetic['benchmarks']:
        for operation in ('encode', 'decode'):
            rows.append(dict(scope='arithmetic_integer_core_cpu', module='arithmetic_'+operation, sample=values['distribution'],
                             baseline_ms=values['baseline_'+operation+'_seconds']*1000,
                             optimized_ms=values['optimized_'+operation+'_seconds']*1000,
                             speedup=values[operation+'_speedup']))
    with (ROOT/'reports/module_speedups.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['scope', 'module', 'sample', 'baseline_ms', 'optimized_ms', 'speedup'])
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    labels = dict(morton3_code='Morton 位交织编码', morton3_sort='初始 Morton 排序与去重',
                  down_once='单层下采样和占用符号', build_hierarchy='完整几何层级构建',
                  upscale_feature='特征上采样', upscale_coordinate='坐标上采样',
                  upscale_coordinate_feature='坐标和特征联合上采样')
    for key, label in labels.items():
        summary.append((label, spread(s['stages'][key]['speedup'] for s in morton['samples']), '四份真实样例，posQ=16'))
    for key, label in [('boe_offline_sort_histograms', 'BoE 离线直方图统计'),
                       ('boe_offline_collect', 'BoE 离线完整直方图收集')]:
        summary.append((label, spread(s['stages'][key]['speedup'] for s in boe['samples']), '中心训练前处理，posQ=4'))
    for key, label in [('PLY read', 'PLY 读取'), ('PLY write', 'PLY 写出')]:
        summary.append((label, spread(s['speedup'] for s in io['results'] if s['module'] == key), '四份真实样例；缓存读写'))
    for operation, label in [('encode', '整数算术编码'), ('decode', '整数算术解码')]:
        summary.append((label, spread(s[operation+'_speedup'] for s in arithmetic['benchmarks'] if s['distribution'].startswith('histogram_')),
                        '真实占用符号＋经验 CDF，非神经预测'))
    for key, n in [('cdf_normalization_64', 64), ('cdf_normalization_4096', 4096), ('cdf_normalization', 100000)]:
        value = next(s for s in modules['rows'] if s['module'] == key)
        summary.append((f'CDF 归一化（N={n}）', spread([value['speedup']]), '17 项 CDF；合成数据'))
    table = '\n'.join(f'| {a} | {b} | {c} |' for a, b, c in summary)
    empirical = [v for v in arithmetic['benchmarks'] if v['distribution'].startswith('histogram_')]
    enc_total = sum(v['baseline_encode_seconds'] for v in empirical)/sum(v['optimized_encode_seconds'] for v in empirical)
    dec_total = sum(v['baseline_decode_seconds'] for v in empirical)/sum(v['optimized_decode_seconds'] for v in empirical)
    concentrated = next(v for v in arithmetic['benchmarks'] if v['distribution'] == 'concentrated')
    detail = '\n'.join(f'| {r["scope"]} | {r["module"]} | {r["sample"].replace(chr(92), "/")} | '
                       f'{r["baseline_ms"]:.4f} | {r["optimized_ms"]:.4f} | {r["speedup"]:.3f}× |' for r in rows)
    document = f'''# ELiCv1 优化、实测与验收报告

更新日期：{REPORT_DATE}（Asia/Hong_Kong）；初始实测日期：2026-10-15。原始版本：`8f3ea78a73d4a3b1ac8d58912fcca10523e4d4db`。

已实现并验证的 CPU 模块中，多项达到数倍至数十倍加速。完整神经网络编解码和 GPU 整机加速仍待验证：本机 CUDA 不可用，缺少 TorchSparse；生产 PyTorch C++ 绑定还因 Windows SDK/UCRT `assert.h` 缺失而无法编译。不能将以下 CPU 模块倍数作为 ELiCv1 整机倍数，也不能据此宣称完整神经 codec 已验收。

本次全部可运行回归：**{passed} 项通过、{subtests} 个子测试通过、{skipped} 项 CUDA 测试跳过，0 失败**。详见 [测试输出](test_output.txt) 和 [JUnit](tests.xml)。完整模型预检结果为 **{pipeline['status']}**，生产绑定构建结果为 **{binding['status']}**，分别保存在 [模型验收状态](pipeline_validation.json) 和 [绑定构建状态](arithmetic_torch_binding.json)。

{memory_io_section(memory_io, experiment, incremental)}## 2026-10-15 整体优化结果（历史实测）

本轮在四份真实点云、共 {overall['total_input_points']:,} 个输入点上，完整几何层级构建的平均耗时由 **{overall['workflows'][0]['baseline_mean_ms']:.3f} ms/帧降至 {overall['workflows'][0]['optimized_mean_ms']:.3f} ms/帧**，汇总加速 **{overall['workflows'][0]['speedup']:.3f}×**。BoE 离线完整直方图收集汇总加速 **{overall['workflows'][1]['speedup']:.3f}×**；PLY 读写合计和经验 CDF 的整数算术编解码合计分别为 **{overall['workflows'][2]['speedup']:.3f}×**、**{overall['workflows'][3]['speedup']:.3f}×**。

以下按完整子流程汇总已有实测记录。先对四帧各分项的计时中位数求和，再除以帧数得到平均耗时；加速比为原版总耗时除以优化总耗时，未对各帧加速比取算术平均。读写、算术编解码由各自分项中位数相加，**不是将这些步骤连起来重新计时的端到端实验**。

| 完整子流程 | 原版平均 ms/帧 | 优化平均 ms/帧 | 汇总加速 | 耗时下降 |
| --- | ---: | ---: | ---: | ---: |
{overall_table}

### 已测 CPU 阶段成本合计（分析值）

仅将「几何层级构建＋单文件 PLY 读取/写出＋经验 CDF 算术编码/解码」相加，四帧成本由 **{budget['baseline_total_ms']:.4f} ms → {budget['optimized_total_ms']:.4f} ms**，折合平均 **{budget['baseline_mean_ms']:.3f} → {budget['optimized_mean_ms']:.3f} ms/帧**，成本比为 **{budget['speedup']:.3f}×**，下降 **{budget['time_reduction_percent']:.2f}%**。该数值只描述这些已测 CPU 工作负载的分析性成本预算，**不是 ELiCv1 完整神经编解码的整体加速比**。

计算只使用完整层级构建一行，不再叠加其排序、位交织和下采样；I/O 只使用每帧读取和写出，不再累计批量读取行。BoE 属于独立的离线训练前处理，且 posQ=4、内部也包含层级构建，因此不并入上述 posQ=16 的成本合计。单模块上采样和合成 CDF 基准没有覆盖完整真实链路的调用次数，也未混入合计。

### 完整 ELiCv1 的结论与待验收指标

**当前能确认的是已测 CPU 子流程明显提速，以及现有可运行的模块/几何/整数码流测试通过；完整神经编解码能否整体达到若干倍，仍无实测结论。** 上述分析合计未包含稀疏卷积与预测、完整解码逐层上采样、实际神经 CDF 生成/归一化、CPU/GPU 传输与同步、模型/JIT 启动等成本，不能据此推导真实 FPS、整机延迟或 GPU 加速比。I/O 写出基准使用原样例坐标，实际解码输出的去重点数也可能不同。

模型架构、权重及预测精度保持原样；完整模型的压缩率、历史神经码流双向兼容、总峰值内存和训练收敛仍待相应环境验证。Windows SDK/UCRT、CUDA 与 TorchSparse 就绪后，使用 `benchmarks/compare_pipeline.py` 对原新版本做完整编解码、逐字节及量化坐标验收，分别报告编码、解码、含读写的整条流程耗时和异步模式收益。

整体汇总数值与计算口径保存于 [JSON](overall_summary.json) 和 [CSV](overall_summary.csv)，可通过 `python benchmarks/write_report.py` 从已有原始记录重新生成，无需重新计时。

## 逐模块实测

| 模块 | 原版耗时 / 优化耗时 | 数据与范围 |
| --- | ---: | --- |
{table}
| 推理时 BoE 选择、稀疏卷积、Embedding、预测网络 | 未测，主要计算保持原样 | 需 CUDA/TorchSparse 完整验收 |
| GPU/CPU 异步熵编码流水 | GPU 收益未测 | 可选 `--async_entropy True`；真实 C++ 核线程顺序测试通过 |

四份样例合计 **412,378** 个输入点。完整层级构建包含初始排序和多层下采样，BoE 完整收集包含其层级构建；这些收益存在重叠，**不能相加或相乘**。BoE 离线前处理结果不代表推理时选择编码器的加速，KMeans、TSNE 未计时、未改变。

算术核在四份经验占用流上的合计耗时比为编码 **{enc_total:.3f}×**、解码 **{dec_total:.3f}×**。极端单峰合成分布的编码为 **{concentrated['encode_speedup']:.3f}×**，约慢 **{(1/concentrated['encode_speedup']-1)*100:.1f}%**；该回退保留披露。CDF 小模块、不同分布及系统负载都会影响倍数，原始计时样本均保存在 JSON。

## 实现与论文对应

1. `src/morton.py`：先按 Morton 整数排序，再相邻去重，去掉二维 unique 的额外排序；位扩张复用临时存储。单层下采样直接用 `code & 7` 得到子八分体，仅对父结点提取坐标。`build_hierarchy` 只读取一次最大位深，去掉逐层主机同步及无用克隆。
2. 特征和坐标上采样按实际占用的 parent/child 索引 gather，避免实体化 `8N×C` 特征和所有候选坐标；联合上采样共享索引。保持 Morton 顺序、坐标与特征对齐；训练特征梯度保留。
3. `src/op.py`：用精确可表示的常量代替每次构造标量 tensor/pow，按 device 缓存 CDF 位置增量；保持 round、int16 和末项 65536 的原约定。
4. `csrc/arithmetic_core.h`：区间状态使用局部变量，CLZ 批量处理 E1/E2/E3 与 bit I/O，常规 CDF 用四次乘法比较替代逐符号除法；极偏 CDF 保留原搜索。10 月 20 日新增相邻 4 字节批量装载和 64 位输入缓存，保留尾部与逻辑字节计数语义。`arithmetic_coding.cpp` 保持 API，并在纯 Tensor Chunk 调用时释放 GIL。
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
| 批量取位缓存（10 月 20 日） | {memory_io['reader_checks']:,} 次直接取位/位置检查、{memory_io['decoder_cases']} 组解码及 {memory_io['geometry_correctness_cases']} 份真实占用流检查；覆盖 0–32 位、流末尾部、截断零填充、分块/reset；符号和已消耗/剩余字节数一致。 |
| 真实几何＋整数熵编码 | 4 份完整样例 × 5 档 posQ=4/8/16/32/64，逐层占用/坐标精确一致；原新编码、四种编解码组合与量化去重目标一致。此项采用由已解坐标和阶段决定的确定性 CDF，**不含训练后神经预测**。 |
| BoE | 冻结原脚本循环 AST 对照，逐层 32 bin、位深标签、跨帧截断顺序、冻结模型的中心选择及平局处理一致。未重新聚类或训练。 |
| I/O | 4 份实际点云原新读入精确一致；额外属性/列重排/其他元素/空单点/损坏与截断/BIN；包含负零、非规格数的 float32 回读；覆盖写出、空格路径与线程顺序。 |
| 异步 | 真实 C++ 整数核经 ctypes 接入生产 EntropyEncoder，分块 FIFO 字节一致、重复 finish/reset；工作线程错误传播。**GPU 重叠和生产 pybind 编译/运行未验证**。 |
| 完整模型与训练 | CUDA/TorchSparse 不可用，未完成真实神经 CDF、原新完整码流双向互解、D1/D2 外部程序、训练收敛或 GPU 端到端时延验收。 |

## 性能测试方法与复现

环境：Windows 11，Intel Family 6 Model 170，22 个逻辑处理器；Python 3.13.12、PyTorch 2.12.0+cpu、NumPy 2.4.2。几何/CDF/BoE 固定 **1 线程**，原版和优化版相同；I/O 默认最多 min(文件数, CPU 数, 8) 个线程，另有顺序对照。基准依次独占运行，预热 2–3 次，历史原新对比交替计时 7/9/11 次；10 月 20 日批量读取增量对比交替计时 15 次，均取中位数。加速比为各自基线与优化耗时中位数之比。这些范围是四个样例的观测范围，非置信区间。

几何与 CDF 比较只计模块调用，输入预先装载；I/O 计打开/解析或格式化/关闭，使用系统缓存、不做 fsync，不代表冷盘速度。原始 occupancy lookup 表只迁移到 CPU，保留其原计算。原 I/O 基线移除了无用 Open3D 构造；批量基线为原单文件读取的顺序调用，因为原 Pool(64) 的 Windows 驱动不可直接运行。算术基线由冻结源码提取，仅替换 Torch/Python 容器接口并去掉 Torch 校验宏，优化核仍检查符号范围；这有利于基线，并未夸大收益。

源码与权重哈希见 [验证清单](validation_manifest.json)。历史模块和样例的原版毫秒、优化毫秒、倍数集中在 [CSV](module_speedups.csv)。历史原始记录分别为 [几何](../benchmarks/results_morton.json)、[BoE](boe_cpu.json)、[I/O](io_benchmark.json)、[CDF/合成](modules_cpu.json)、[算术核](arithmetic_benchmark.json)；本轮批量读取增量对比单列于 [JSON](memory_io_benchmark.json) 和 [CSV](memory_io_benchmark.csv)。

完整命令与输入单位说明见 [复现指南](../OPTIMIZATION.md)。重点入口：

```powershell
$elicPython = 'C:\\Users\\LinkZelda\\miniconda3\\python.exe'
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
{detail}
'''
    (ROOT/'reports/optimization_report.md').write_text(document, encoding='utf-8', newline='\n')
    code_paths = [ROOT/p for p in ['BoE_cluster.py', 'compress.py', 'decompress.py', 'model.py', 'src/boe.py',
                  'src/morton.py', 'src/io.py', 'src/op.py', 'src/runtime.py', 'src/nn.py',
                  'csrc/arithmetic_core.h', 'csrc/arithmetic_coding.cpp']]
    manifest = dict(date=REPORT_DATE, timezone='Asia/Hong_Kong',
                    baseline_commit='8f3ea78a73d4a3b1ac8d58912fcca10523e4d4db',
                    code_sha256={str(p.relative_to(ROOT)).replace(chr(92), '/'): digest(p) for p in code_paths},
                    weights_sha256={str(p.relative_to(ROOT)).replace(chr(92), '/'): digest(p) for p in sorted((ROOT/'model').glob('*/*.pt'))},
                    inputs_sha256={str(p.relative_to(ROOT)).replace(chr(92), '/'): digest(p) for p in sorted((ROOT/'data').glob('*Example/*.ply'))},
                    baseline_manifest_sha256=digest(ROOT/'benchmarks/reference/manifest.json'),
                    pytest=dict(passed=passed, subtests_passed=subtests, skipped_cuda=skipped, failed=0),
                    full_neural_pipeline_status=pipeline['status'], production_torch_binding_status=binding['status'])
    manifest['memory_io_experiment'] = dict(
        date=REPORT_DATE, baseline_kind=incremental['baseline_kind'],
        baseline_core_sha256=memory_io['baseline_sha256'], optimized_core_sha256=memory_io['optimized_sha256'],
        benchmark='reports/memory_io_benchmark.json', benchmark_sha256=digest(ROOT/'reports/memory_io_benchmark.json'),
        benchmark_csv_sha256=digest(ROOT/'reports/memory_io_benchmark.csv'),
        source_record=f'{MEMORY_IO_DIR}/bitreader_final.json',
        reader_checks=memory_io['reader_checks'], decoder_cases=memory_io['decoder_cases'],
        real_geometry_cases=memory_io['geometry_correctness_cases'], original_compatibility_cases=447,
        pytest_log_sha256=digest(ROOT/'reports/test_output.txt'), pytest_junit_sha256=digest(ROOT/'reports/tests.xml'),
        rejected_tensor_trial=f'{MEMORY_IO_DIR}/trials/tensor_batch/tensor_benchmark.json')
    manifest['historical_arithmetic_measurement'] = dict(
        date='2026-10-15', source='reports/arithmetic_benchmark.json',
        optimized_core_sha256=arithmetic['core_sha256'])
    manifest['benchmark_sha256'] = {path: digest(ROOT/path) for path in (
        'benchmarks/bench_bitreader_batch.py', 'benchmarks/bitreader_batch_test.cpp', 'benchmarks/write_report.py')}
    (ROOT/'reports/validation_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Report generated: {len(rows)} module/sample timing rows; {passed} tests + {subtests} subtests passed; {skipped} CUDA skips.')


if __name__ == '__main__':
    main()
