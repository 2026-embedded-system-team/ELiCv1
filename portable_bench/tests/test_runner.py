"""Contracts whose failure would mislabel correctness, timing or devices."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

TOOL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL))
from elic_bench.backend import Backend, BackendUnavailable
from elic_bench.cli import main
from elic_bench.config import ConfigError, normalize_config
from elic_bench.context import Context
from elic_bench.results import summarize, write_outputs


class ConfigurationTests(unittest.TestCase):
    def test_reject_invalid_config(self):
        for invalid in ({"repeats": 0}, {"threads": True}, {"points": -1}, {"posq": [16, 16]},
                        {"device": "mps"}, {"suites": []}, {"pretend_cuda": True},
                        {"repo_root": 123}, {"output_dir": ["invalid"]}):
            with self.subTest(config=invalid), self.assertRaises(ConfigError):
                normalize_config(invalid)

    def test_cli_config_path_and_overrides(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            cloud = folder / "cloud.ply"
            cloud.write_text("fixture", encoding="ascii")
            source = folder / "test.json"
            source.write_text(json.dumps(dict(files=["cloud.ply"], synthetic=True, correctness_only=True)), encoding="utf-8")
            captured = {}
            def fake_run(values):
                captured.update(normalize_config(values))
                return dict(summary={"status": "passed", "exit_code": 0}, config=captured, backend=None)
            with patch("elic_bench.cli.run_benchmarks", side_effect=fake_run), contextlib.redirect_stdout(io.StringIO()):
                code = main(["run", "--config", str(source), "--no-synthetic", "--no-correctness-only", "--prequantized", "yes"])
            self.assertEqual(code, 0)
            self.assertEqual(captured["files"][0]["path"], str(cloud.resolve()))
            self.assertTrue(captured["files"][0]["prequantized"])
            self.assertFalse(captured["correctness_only"])

    def test_custom_units_and_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "input.bin").write_bytes(b"")
            config = normalize_config(dict(files=[dict(path="input.bin", prequantized=False, scale=.01, offset=10)]), base_dir=folder)
            self.assertEqual(config["files"][0]["scale"], .01)
            self.assertEqual(config["files"][0]["offset"], 10)


class DeviceTests(unittest.TestCase):
    def test_simulation_does_not_call_cuda_synchronize_or_set_device(self):
        calls = []
        fake_torch = SimpleNamespace(device=lambda name: SimpleNamespace(type=name),
                                     cuda=SimpleNamespace(is_available=lambda: False,
                                                          synchronize=lambda *a: calls.append("sync"),
                                                          set_device=lambda *a: calls.append("set")))
        backend = Backend("cuda-sim", fake_torch)
        backend.synchronize()
        self.assertTrue(backend.simulated)
        self.assertEqual(backend.device.type, "cpu")
        self.assertEqual(calls, [])

    def test_explicit_cuda_never_silently_falls_back(self):
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with self.assertRaises(BackendUnavailable):
            Backend("cuda", fake_torch)


class ContextTests(unittest.TestCase):
    def ctx(self, simulated=False):
        config = normalize_config(dict(synthetic=True, warmups=0, repeats=3))
        backend = SimpleNamespace(device=SimpleNamespace(type="cpu"), simulated=simulated,
                                  synchronize=lambda: None)
        return Context(config, backend)

    def test_false_checker_is_failure_without_measurement(self):
        ctx = self.ctx()
        case = ctx.record_case("bad", "custom", lambda: 1, lambda: 2, check=lambda a, b: False)
        self.assertEqual(case["status"], "failed")
        self.assertIsNone(case["timing"])
        self.assertEqual(summarize(ctx.records)["exit_code"], 1)

    def test_paired_samples_and_simulation_null_timing(self):
        ctx = self.ctx()
        case = ctx.record_case("good", "custom", lambda: 1, lambda: 1)
        self.assertEqual(len(case["timing"]["baseline"]["samples_ms"]), 3)
        sim = self.ctx(simulated=True)
        case = sim.record_case("sim", "custom", lambda: 1, lambda: 1, details={"uses_cuda": True})
        self.assertEqual(case["execution_kind"], "interface_simulation")
        self.assertIsNone(case["timing"])
        self.assertFalse(case["cuda_execution_validated"])

    def test_timing_exception_is_error_and_not_counted_as_correct(self):
        ctx = self.ctx()
        state = {"calls": 0}
        def broken():
            state["calls"] += 1
            if state["calls"] > 1:
                raise RuntimeError("fails in timed call")
            return 1
        case = ctx.record_case("error", "custom", lambda: 1, broken)
        self.assertEqual(case["status"], "error")
        self.assertIsNone(summarize(ctx.records)["correctness"]["pass_rate"])

    def test_junit_and_csv_keep_partial_coverage_visible(self):
        ctx = self.ctx()
        ctx.record_case("yes", "custom", lambda: 1, lambda: 1, details={"timing_scope": "correctness_only"})
        ctx.record_case("no", "custom", lambda: 1, lambda: 2)
        ctx.add_result("missing", "custom", "unavailable", "no compiler")
        report = dict(schema_version="1.0", cases=ctx.records, summary=summarize(ctx.records))
        self.assertEqual(report["summary"]["correctness"]["pass_rate"], .5)
        self.assertFalse(report["summary"]["coverage_complete"])
        with tempfile.TemporaryDirectory() as temporary:
            write_outputs(report, temporary)
            suite = ET.parse(Path(temporary) / "junit.xml").getroot()
            self.assertEqual(suite.get("failures"), "1")
            self.assertEqual(suite.get("skipped"), "1")
            self.assertEqual(len(suite.findall("testcase")), 3)


class BindingClassificationTests(unittest.TestCase):
    def test_build_regressions_are_errors_and_missing_sdk_is_unavailable(self):
        from elic_bench.external import run_binding
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            config = normalize_config(dict(output_dir=str(folder), synthetic=True))
            backend = SimpleNamespace(device=SimpleNamespace(type="cpu"), simulated=False)
            for index, (reason, expected) in enumerate((
                    ("Torch extension build failed: syntax error C2143 in arithmetic_coding.cpp", "error"),
                    ("Torch extension build failed: Cannot open include file: 'assert.h': No such file", "unavailable"),
                    ("Torch binding smoke test failed: round trip mismatch", "failed"))):
                stage = folder / str(index)
                stage.mkdir()
                def fake_process(*args, **kwargs):
                    (stage / "binding_result.json").write_text(json.dumps(dict(status="unavailable", reason=reason)), encoding="utf-8")
                    return 1, .01
                ctx = Context(config, backend)
                with patch("elic_bench.external._stage", return_value=stage), patch("elic_bench.external.run_process", side_effect=fake_process):
                    result = run_binding(ctx)
                self.assertEqual(result["status"], expected)


if __name__ == "__main__":
    unittest.main()
