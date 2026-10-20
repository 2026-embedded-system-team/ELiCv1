"""Quantization checks at boundaries where a dtype change alters geometry."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from elic_bench.backend import Backend
from elic_bench.config import normalize_config
from elic_bench.context import Context
from elic_bench.suites import geometry_inputs, load_modules, run_io
from elic_bench.config import default_repo_root


REPO = default_repo_root()


class TestGeometryQuantization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, modules = load_modules(REPO)
        cls.io = modules.io

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="elic-quantization-test-")
        self.folder = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def write_ply(self, values):
        path = self.folder / "rounding boundary.ply"
        header = ("ply\nformat ascii 1.0\nelement vertex " + str(len(values)) +
                  "\nproperty double x\nproperty double y\nproperty double z\nend_header\n")
        with path.open("w", encoding="ascii") as output:
            output.write(header)
            np.savetxt(output, values, fmt="%.17g")
        return path

    def quantize(self, path, prequantized):
        ctx = SimpleNamespace(repo_root=REPO, device=torch.device("cpu"),
            config={"files": [{"path": str(path), "prequantized": prequantized,
                               "scale": .001, "offset": 131072}], "posq": [4]})
        _, coords, metadata = next(geometry_inputs(ctx, self.io))
        return coords[:, 1:], metadata

    @staticmethod
    def production_quantize(points, prequantized):
        if prequantized:
            xyz = torch.tensor(points)
            if xyz.min().item() < 0:
                xyz += 131072
        else:
            xyz = torch.tensor(points / .001 + 131072)
        return torch.round(xyz / 4).int()

    def test_ply_float64_metres_keep_values_above_rounding_tie(self):
        path = self.write_ply(np.array([[.002000001, 0, -.002000001]], dtype=np.float64))
        points = self.io.read_points(path)
        actual, metadata = self.quantize(path, False)
        expected = self.production_quantize(points, False)
        narrowed = self.production_quantize(points.astype(np.float32), False)
        self.assertFalse(torch.equal(expected, narrowed), "Fixture must expose float32 narrowing")
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(metadata["input_dtype"], "float64")

    def test_ply_prequantized_negative_shift_preserves_float64_boundary(self):
        path = self.write_ply(np.array([[-131069.9999999, 0, 4]], dtype=np.float64))
        points = self.io.read_points(path)
        actual, metadata = self.quantize(path, True)
        expected = self.production_quantize(points, True)
        narrowed = self.production_quantize(points.astype(np.float32), True)
        self.assertFalse(torch.equal(expected, narrowed), "Fixture must expose float32 narrowing")
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(metadata["input_dtype"], "float64")

    def test_binary_xyzi_keeps_numpy_float32_scaling_semantics(self):
        path = self.folder / "rounding boundary.bin"
        records = np.array([[.002000001, -.002000001, .00199999, .25],
                            [1.235, -2.5, 4, .75]], dtype=np.float32)
        records.tofile(path)
        points = self.io.read_points(path)
        original = points.copy()
        actual, metadata = self.quantize(path, False)
        self.assertTrue(torch.equal(self.production_quantize(points, False), actual))
        self.assertEqual(metadata["input_dtype"], "float32")
        np.testing.assert_array_equal(points, original)


class TestCustomIOOracle(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="elic-custom-io-test-")
        self.folder = Path(self.temporary.name)
        self.ply = self.folder / "named XYZ with faces.ply"
        self.ply.write_text(
            "ply\nformat ascii 1.0\nelement edge 1\nproperty int source\nproperty int target\n"
            "element vertex 2\nproperty uchar red\nproperty float z\nproperty double x\nproperty float y\n"
            "element face 1\nproperty list uchar int vertex_indices\nend_header\n"
            "0 1\n255\t3 1 2\n10 6 4 5\n3 0 1 0\n", encoding="utf-8")
        self.binary = self.folder / "uppercase.BIN"
        np.array([[1.25, -2.5, 3.75, .5], [9, 8, 7, 6]], dtype=np.float32).tofile(self.binary)

    def tearDown(self):
        self.temporary.cleanup()

    def context(self, correctness_only=False):
        config = normalize_config(dict(repo_root=str(REPO), output_dir=str(self.folder / "output"),
            points=16, suites=["io"], correctness_only=correctness_only,
            files=[{"path": str(self.ply), "prequantized": True},
                   {"path": str(self.binary), "prequantized": True}]))
        return Context(config, Backend("cpu", torch))

    def test_valid_custom_formats_use_oracle_without_baseline_timing(self):
        # Global timing is enabled: these two individual cases must disable it
        # themselves because the frozen reader cannot parse these valid files.
        context = self.context()
        results = run_io(context)
        try:
            records = [next(results), next(results)]
        finally:
            results.close()
        for record in records:
            self.assertEqual(record["status"], "passed", record.get("traceback"))
            self.assertIsNone(record["timing"])
            self.assertEqual(record["details"]["timing_scope"], "correctness_only")
            self.assertIn("independent file-format oracle", record["details"]["baseline"])
            self.assertIn("frozen reader unsupported", record["details"]["baseline"])

    def test_batch_with_custom_formats_uses_ordered_oracles(self):
        context = self.context(correctness_only=True)
        records = list(run_io(context))
        self.assertTrue(all(row["status"] == "passed" for row in records), records)
        batches = [row for row in records if row["case_id"].startswith("io.batch_read.")]
        self.assertEqual(len(batches), 2)
        for record in batches:
            self.assertIsNone(record["timing"])
            self.assertEqual(record["details"]["timing_scope"], "correctness_only")
            self.assertIn("frozen reader unsupported", record["details"]["baseline"])

    def test_candidate_value_error_is_detected_by_independent_oracle(self):
        _, modules = load_modules(REPO)
        original_reader = modules.io.read_points
        def wrong_read(path):
            points = original_reader(path).copy()
            points[:, 0] += 1
            return points
        bad_io = SimpleNamespace(read_points=wrong_read,
            save_ply_ascii_geo=modules.io.save_ply_ascii_geo,
            read_point_clouds=modules.io.read_point_clouds)
        context = self.context()
        with mock.patch("elic_bench.suites.load_modules", return_value=(None, SimpleNamespace(io=bad_io))):
            results = run_io(context)
            try:
                record = next(results)
            finally:
                results.close()
        self.assertEqual(record["status"], "failed")
        self.assertFalse(record["correctness"]["check_passed"])
        self.assertIsNone(record["timing"])


if __name__ == "__main__":
    unittest.main()
