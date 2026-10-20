"""Extension contract and paired synchronized host-latency timing."""
import statistics
import time
import traceback


def exact_equal(before, after):
    if isinstance(before, (list, tuple)):
        assert type(before) is type(after) and len(before) == len(after), "Sequence structure differs"
        for left, right in zip(before, after):
            exact_equal(left, right)
    elif isinstance(before, dict):
        assert before.keys() == after.keys(), "Dictionary keys differ"
        for key in before:
            exact_equal(before[key], after[key])
    elif hasattr(before, "detach"):
        import torch
        assert before.shape == after.shape and before.dtype == after.dtype and torch.equal(before, after), "Tensor shape, dtype or values differ"
    elif hasattr(before, "dtype"):
        import numpy as np
        assert before.shape == after.shape and before.dtype == after.dtype and np.array_equal(before, after), "Array shape, dtype or values differ"
    else:
        assert before == after, "Values differ"


def stats(samples):
    values = sorted(samples)
    def percentile(fraction):
        position = (len(values)-1)*fraction
        low = int(position)
        high = min(low+1, len(values)-1)
        return values[low] + (values[high]-values[low])*(position-low)
    return dict(median_ms=statistics.median(values), p10_ms=percentile(.1), p90_ms=percentile(.9),
                min_ms=values[0], max_ms=values[-1], samples_ms=samples)


class Context:
    def __init__(self, config, backend):
        from pathlib import Path
        self.config, self.backend = config, backend
        self.repo_root = Path(config["repo_root"])
        self.output_dir = Path(config["output_dir"])
        self.device, self.simulated = backend.device, backend.simulated
        self.records = []

    def add_result(self, case_id, suite, status, reason=None, **extra):
        if any(r["case_id"] == case_id for r in self.records):
            raise ValueError(f"Duplicate case ID: {case_id}")
        case = dict(case_id=case_id, suite=suite, status=status,
                    execution_kind="interface_simulation" if self.simulated else "hardware",
                    cuda_execution_validated=False, correctness=dict(check_passed=None), timing=None,
                    **extra)
        if reason:
            case["reason"] = str(reason)
        self.records.append(case)
        return case

    def record_case(self, case_id, suite, before, after, check=None, details=None):
        details = dict(details or {})
        case = self.add_result(case_id, suite, "error", details=details)
        started = time.perf_counter()
        try:
            left, right = before(), after()
            metrics = (check or exact_equal)(left, right)
            if metrics is False:
                raise AssertionError("Correctness checker returned False")
            del left, right
            case["correctness"] = dict(check_passed=True, metrics=metrics if isinstance(metrics, dict) else {})
            case["status"] = "passed"
            case["cuda_execution_validated"] = bool(not self.simulated and self.device.type == "cuda" and details.get("uses_cuda"))
            disabled = self.simulated or self.config["correctness_only"] or details.get("timing_scope") == "correctness_only"
            if disabled:
                case["timing_disabled_reason"] = "interface_simulation" if self.simulated else "correctness_only"
                return case
            functions = (before, after)
            for iteration in range(self.config["warmups"]):
                for index in ((0, 1) if iteration % 2 == 0 else (1, 0)):
                    result = functions[index]()
                    self.backend.synchronize()
                    del result
            samples = [[], []]
            for iteration in range(self.config["repeats"]):
                for index in ((0, 1) if iteration % 2 == 0 else (1, 0)):
                    self.backend.synchronize()
                    tick = time.perf_counter_ns()
                    result = functions[index]()
                    self.backend.synchronize()
                    samples[index].append((time.perf_counter_ns()-tick)/1e6)
                    del result
            baseline, candidate = map(stats, samples)
            case["timing"] = dict(unit="ms", clock="perf_counter_ns", aggregation="median",
                                  order="alternating baseline/candidate", warmups=self.config["warmups"],
                                  repeats=self.config["repeats"], baseline=baseline, candidate=candidate,
                                  cuda_synchronized=self.device.type == "cuda",
                                  speedup=baseline["median_ms"]/candidate["median_ms"] if candidate["median_ms"] else None,
                                  timing_scope=details.get("timing_scope", "operation call including returned-output allocation"))
        except AssertionError as error:
            case.update(status="failed", reason=str(error) or "Correctness assertion failed",
                        correctness=dict(check_passed=False), timing=None, cuda_execution_validated=False,
                        traceback=traceback.format_exc())
        except Exception as error:
            case.update(status="error", reason=f"{type(error).__name__}: {error}", timing=None,
                        cuda_execution_validated=False, traceback=traceback.format_exc())
        finally:
            case["case_wall_seconds"] = time.perf_counter()-started
        return case
