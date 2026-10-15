"""Pure-torch regression checks for offline BoE histogram preprocessing."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))
from src import boe, morton
from bench_boe import load_frozen_functions


class BoEHistogramTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.original_collect, cls.original_histogram, cls.original_select = tuple(
            staticmethod(function) for function in load_frozen_functions())

    def assert_tensor_equal(self, result, baseline):
        self.assertEqual(result.dtype, baseline.dtype)
        self.assertEqual(result.device, baseline.device)
        self.assertTrue(torch.equal(result, baseline))

    def test_all_256_symbols_and_permutation_invariance(self):
        occupancy = torch.arange(256, dtype=torch.int32)
        coords = torch.randint(0, 200, (256, 4), generator=torch.Generator().manual_seed(8),
                               dtype=torch.int32)
        expected = self.original_histogram(coords, occupancy)
        self.assert_tensor_equal(boe.occupancy_histogram(occupancy), expected)
        self.assert_tensor_equal(expected, torch.full((32,), 1 / 32))
        order = torch.randperm(256, generator=torch.Generator().manual_seed(20))
        self.assert_tensor_equal(boe.occupancy_histogram(occupancy[order]), expected)
        for dtype in (torch.int16, torch.int32, torch.int64):
            self.assert_tensor_equal(boe.occupancy_histogram(occupancy.to(dtype).view(-1, 1)), expected)

    def test_random_hierarchies_match_frozen_script_with_legacy_labels(self):
        gen = torch.Generator().manual_seed(77)
        for depth in (0, 1, 3, 7, 8, 12, 16, 21):
            for dtype in (torch.int32, torch.int64):
                with self.subTest(depth=depth, dtype=dtype):
                    xyz = torch.randint(0, 1 << depth, (250, 3), generator=gen, dtype=dtype)
                    coords = torch.cat((torch.zeros((250, 1), dtype=dtype), xyz), dim=1)
                    coords = torch.cat((coords, coords[:25]), dim=0)
                    original = coords.clone()
                    actual_h, actual_b = boe.collect_histograms(coords)
                    expected_h, expected_b = self.original_collect(coords)
                    self.assert_tensor_equal(actual_h, expected_h)
                    self.assert_tensor_equal(actual_b, expected_b)
                    self.assertEqual(int(actual_b[0, 0]), 7)
                    self.assert_tensor_equal(actual_b.view(-1), torch.arange(7, 7 + len(actual_b), dtype=torch.int32))
                    self.assert_tensor_equal(coords, original)

    def test_sample_truncation_keeps_frame_and_coarse_to_fine_order(self):
        gen = torch.Generator().manual_seed(13)
        coordinates = []
        for depth in (12, 3, 16):
            xyz = torch.randint(0, 1 << depth, (150, 3), generator=gen, dtype=torch.int32)
            coordinates.append(torch.cat((torch.zeros((150, 1), dtype=torch.int32), xyz), dim=1))
        for limit in (1, 4, 5, 6, 9, 12):
            optimized = [boe.collect_histograms(c) for c in coordinates]
            actual_h = torch.cat([h for h, _ in optimized])[:limit]
            actual_b = torch.cat([b for _, b in optimized])[:limit]
            old_h, old_b, remaining = [], [], limit
            for coords in coordinates:
                histograms, labels = self.original_collect(coords, max_samples=remaining)
                old_h.append(histograms)
                old_b.append(labels)
                remaining -= len(histograms)
                if remaining == 0:
                    break
            self.assert_tensor_equal(actual_h, torch.cat(old_h))
            self.assert_tensor_equal(actual_b, torch.cat(old_b))

    def test_center_selection_indices_and_ties_match_frozen_model(self):
        gen = torch.Generator().manual_seed(18)
        xyz = torch.randint(0, 1 << 14, (350, 3), generator=gen, dtype=torch.int32)
        coords = torch.cat((torch.zeros((350, 1), dtype=torch.int32), xyz), dim=1)
        histograms, _ = boe.collect_histograms(coords)
        hierarchy = morton.build_hierarchy(coords, stop_bitdepth=7)
        centers = torch.rand((5, 32), generator=gen)
        centers /= centers.sum(dim=1, keepdim=True)
        for histogram, (_, occupancy) in zip(histograms, hierarchy):
            for candidates in (centers, torch.stack((histogram, histogram, histogram + 1))):
                expected = self.original_select(SimpleNamespace(centers=candidates), occupancy & 15, occupancy >> 4)
                actual = int(torch.norm(candidates - histogram[None, :], dim=1).argmin())
                self.assertEqual(actual, expected)
                if len(candidates) == 3:
                    self.assertEqual(actual, 0)

    def test_empty_inputs(self):
        histogram = boe.occupancy_histogram(torch.empty(0, dtype=torch.int32))
        self.assert_tensor_equal(histogram, torch.zeros(32))
        histograms, labels = boe.collect_histograms(torch.empty((0, 4), dtype=torch.int32))
        self.assertEqual(tuple(histograms.shape), (0, 32))
        self.assertEqual(histograms.dtype, torch.float32)
        self.assertEqual(tuple(labels.shape), (0, 1))
        self.assertEqual(labels.dtype, torch.int32)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_statistics_match_frozen_script(self):
        xyz = torch.randint(0, 1 << 14, (500, 3), dtype=torch.int32, device="cuda")
        coords = torch.cat((xyz[:, :1] * 0, xyz), dim=1)
        actual, expected = boe.collect_histograms(coords), self.original_collect(coords)
        for result, reference in zip(actual, expected):
            self.assert_tensor_equal(result, reference)


if __name__ == "__main__":
    unittest.main()
