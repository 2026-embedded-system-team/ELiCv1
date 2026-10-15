"""Exact geometry/occupancy and autograd checks against the frozen original.

Run: python -m unittest discover -s tests -p test_morton.py -v
"""
import importlib.util
import math
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import morton as mt

spec = importlib.util.spec_from_file_location(
    "reference_morton", ROOT / "benchmarks/reference/src/morton.py")
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
# The original unconditionally defaults these tables to CUDA. Initializing
# those unchanged tables on CPU lets us compare its actual tensor algorithm.
reference._get_occ_bits(torch.device("cpu"))
reference._get_expand_coords_base(torch.device("cpu"))


def reference_hierarchy(coords, stop_bitdepth=2):
    coords, code = reference.morton3_sort(coords)
    hierarchy = []
    while True:
        coords, code, occupancy = reference.down_once(coords, code)
        hierarchy.append((coords.clone(), occupancy.clone()))
        bitdepth = math.ceil(math.log2(int(coords[:, 1:].max()) + 1))
        if bitdepth <= stop_bitdepth:
            break
    return hierarchy[::-1]


class MortonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Fixed thread count also avoids very large thread-pool overhead on
        # small CPU geometry tests. Timing is done by the benchmark script.
        torch.set_num_threads(1)

    def assert_tensors_equal(self, actual, expected):
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.device, expected.device)
        self.assertTrue(torch.equal(actual, expected),
                        f"shape/value mismatch: {actual.shape} vs {expected.shape}")

    def test_codes_include_21_bit_boundary(self):
        xyz = torch.tensor([[0, 0, 0], [1, 2, 4], [(1 << 21) - 1] * 3,
                            [1 << 20, 0, 0], [0, 1 << 20, 0], [0, 0, 1 << 20]])
        expected = []
        for row in xyz.tolist():
            expected.append(sum(((value >> bit) & 1) << (3 * bit + axis)
                                for axis, value in enumerate(row)
                                for bit in range(21)))
        self.assert_tensors_equal(mt.morton3_code(xyz), torch.tensor(expected))
        self.assert_tensors_equal(mt.morton3_code(xyz), reference.morton3_code(xyz))

    def test_sort_and_every_down_level_match_reference(self):
        gen = torch.Generator().manual_seed(1234)
        for dtype in (torch.int32, torch.int64):
            for depth, count in ((0, 1), (1, 8), (3, 200), (10, 3000), (21, 1000)):
                with self.subTest(dtype=dtype, depth=depth):
                    xyz = torch.randint(0, 1 << depth, (count, 3),
                                        generator=gen, dtype=dtype)
                    coords = torch.cat((torch.zeros(count, 1, dtype=dtype), xyz), dim=1)
                    coords = torch.cat((coords, coords[::3], coords[:1]), dim=0)
                    original = coords.clone()
                    actual_c, actual_code = mt.morton3_sort(coords)
                    expected_c, expected_code = reference.morton3_sort(coords)
                    self.assert_tensors_equal(actual_c, expected_c)
                    self.assert_tensors_equal(actual_code, expected_code)
                    for _ in range(max(1, depth)):
                        actual = mt.down_once(actual_c, actual_code)
                        expected = reference.down_once(expected_c, expected_code)
                        for result, baseline in zip(actual, expected):
                            self.assert_tensors_equal(result, baseline)
                        actual_c, actual_code, _ = actual
                        expected_c, expected_code, _ = expected
                    self.assert_tensors_equal(coords, original)

    def test_all_256_occupancies_match_reference_and_round_trip(self):
        occupancy = torch.arange(256, dtype=torch.int32)
        coords = torch.zeros((256, 4), dtype=torch.int32)
        coords[:, 1] = torch.arange(256, dtype=torch.int32)
        features = torch.arange(256 * 7, dtype=torch.float32).reshape(256, 7)
        children = mt.upscale_coordinate(coords, occupancy)
        self.assert_tensors_equal(children, reference.upscale_coordinate(coords, occupancy))
        expected_features = reference.upscale_feature(occupancy, features)
        self.assert_tensors_equal(mt.upscale_feature(occupancy, features), expected_features)
        child_c, child_f = mt.upscale_coordinate_feature(coords, occupancy, features)
        expected_c, expected_f = reference.upscale_coordinate_feature(coords, occupancy, features)
        self.assert_tensors_equal(child_c, expected_c)
        self.assert_tensors_equal(child_f, expected_f)
        # Occupancy zero emits no parent; the other 255 symbols reconstruct.
        children, codes = mt.morton3_sort(children)
        parent_c, _, parent_o = mt.down_once(children, codes)
        self.assert_tensors_equal(parent_c, coords[1:])
        self.assert_tensors_equal(parent_o, occupancy[1:])

    def test_hierarchy_matches_reference_and_reconstructs(self):
        gen = torch.Generator().manual_seed(2468)
        for depth in (0, 1, 2, 3, 7, 12, 21):
            for stop in (0, 2, 7):
                with self.subTest(depth=depth, stop=stop):
                    xyz = torch.randint(0, 1 << depth, (500, 3), generator=gen,
                                        dtype=torch.int32)
                    coords = torch.cat((torch.zeros(500, 1, dtype=torch.int32), xyz), dim=1)
                    hierarchy = mt.build_hierarchy(coords, stop)
                    expected = reference_hierarchy(coords, stop)
                    self.assertEqual(len(hierarchy), len(expected))
                    for (a_c, a_o), (e_c, e_o) in zip(hierarchy, expected):
                        self.assert_tensors_equal(a_c, e_c)
                        self.assert_tensors_equal(a_o, e_o)
                    for level, (a_c, a_o) in enumerate(hierarchy):
                        expected_c = (hierarchy[level + 1][0] if level + 1 < len(hierarchy)
                                      else mt.morton3_sort(coords)[0])
                        self.assert_tensors_equal(mt.upscale_coordinate(a_c, a_o), expected_c)

    def test_empty_input(self):
        coords = torch.empty((0, 4), dtype=torch.int32)
        features = torch.empty((0, 11), requires_grad=True)
        occupancy = torch.empty(0, dtype=torch.int32)
        sorted_c, code = mt.morton3_sort(coords)
        self.assertEqual(tuple(sorted_c.shape), (0, 4))
        self.assertEqual(tuple(code.shape), (0,))
        parent_c, parent_code, parent_o = mt.down_once(sorted_c, code)
        self.assertEqual(tuple(parent_c.shape), (0, 4))
        self.assertEqual(parent_code.numel(), 0)
        self.assertEqual(parent_o.numel(), 0)
        self.assertEqual(mt.build_hierarchy(coords), [])
        self.assertEqual(tuple(mt.upscale_coordinate(coords, occupancy).shape), (0, 4))
        combined_c, combined_f = mt.upscale_coordinate_feature(coords, occupancy, features)
        self.assertEqual(tuple(combined_c.shape), (0, 4))
        self.assertEqual(tuple(combined_f.shape), (0, 11))
        combined_f.sum().backward()
        self.assertEqual(tuple(features.grad.shape), (0, 11))

    def test_feature_gradient_matches_reference_and_independent_counts(self):
        occupancy = torch.arange(256, dtype=torch.int32)
        expected_counts = torch.tensor([int(value).bit_count() for value in range(256)])
        values = torch.randn((256, 9), generator=torch.Generator().manual_seed(8),
                             dtype=torch.float64)
        for combined in (False, True):
            with self.subTest(combined=combined):
                features = values.clone().requires_grad_()
                if combined:
                    coords = torch.zeros((256, 4), dtype=torch.int32)
                    _, output = mt.upscale_coordinate_feature(coords, occupancy, features)
                else:
                    output = mt.upscale_feature(occupancy, features)
                output.square().sum().backward()
                expected_grad = 2 * values * expected_counts[:, None]
                torch.testing.assert_close(features.grad, expected_grad, rtol=1e-14, atol=1e-14)
                reference_features = values.clone().requires_grad_()
                reference.upscale_feature(occupancy, reference_features).square().sum().backward()
                # index_select accumulates repeated-index gradients in a
                # different addition order than repeat+mask's backward.
                torch.testing.assert_close(features.grad, reference_features.grad,
                                           rtol=1e-14, atol=1e-14)

    def test_multibatch_explicit_mode_preserves_ids(self):
        # Same spatial voxels appear in several batches; duplicate rows must
        # be removed within each batch, never across batches.
        coords = torch.tensor([[7, 3, 1, 0], [-2, 0, 0, 0], [7, 0, 0, 0],
                               [-2, 3, 1, 0], [7, 3, 1, 0], [99, 0, 0, 0]],
                              dtype=torch.int32)
        sorted_c, code = mt.morton3_sort(coords, batch_mode="multiple")
        expected = torch.tensor([[-2, 0, 0, 0], [-2, 3, 1, 0], [7, 0, 0, 0],
                                 [7, 3, 1, 0], [99, 0, 0, 0]], dtype=torch.int32)
        self.assert_tensors_equal(sorted_c, expected)
        parent_c, _, parent_o = mt.down_once(sorted_c, code)
        self.assert_tensors_equal(mt.upscale_coordinate(parent_c, parent_o), sorted_c)
        hierarchy = mt.build_hierarchy(coords, stop_bitdepth=0, batch_mode="multiple")
        for level, (c, occupancy) in enumerate(hierarchy):
            expected_c = hierarchy[level + 1][0] if level + 1 < len(hierarchy) else sorted_c
            self.assert_tensors_equal(mt.upscale_coordinate(c, occupancy), expected_c)
        # The single-batch path also preserves a nonzero batch ID.
        single = coords[coords[:, 0] == 7]
        hierarchy = mt.build_hierarchy(single)
        self.assertTrue(bool(torch.all(hierarchy[0][0][:, 0] == 7)))

    def test_tables_are_cached_on_requested_device(self):
        for device in (torch.device("cpu"), "cpu"):
            self.assertEqual(mt._get_occ_bits(device).device, torch.device("cpu"))
            self.assertEqual(mt._get_expand_coords_base(device).device, torch.device("cpu"))
        self.assertIs(mt._get_occ_bits("cpu"), mt._get_occ_bits(torch.device("cpu")))
        if not torch.cuda.is_available():
            self.assertEqual(mt._get_occ_bits().device, torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_round_trip_and_cache_isolation(self):
        cpu_table = mt._get_occ_bits("cpu")
        for ordinal in range(torch.cuda.device_count()):
            device = torch.device("cuda", ordinal)
            coords = torch.tensor([[0, 1, 2, 3], [0, 7, 6, 5]], dtype=torch.int32, device=device)
            sorted_c, code = mt.morton3_sort(coords)
            parent_c, _, occupancy = mt.down_once(sorted_c, code)
            self.assert_tensors_equal(mt.upscale_coordinate(parent_c, occupancy), sorted_c)
            self.assertEqual(mt._get_occ_bits(device).device, device)
            self.assertIs(mt._get_occ_bits("cpu"), cpu_table)

    def test_invalid_interface_arguments(self):
        with self.assertRaises(ValueError):
            mt.morton3_sort(torch.zeros((2, 3), dtype=torch.int32))
        with self.assertRaises(TypeError):
            mt.morton3_sort(torch.zeros((2, 4)))
        with self.assertRaises(ValueError):
            mt.morton3_sort(torch.zeros((2, 4), dtype=torch.int32), batch_mode="auto")
        with self.assertRaises(ValueError):
            mt.build_hierarchy(torch.zeros((2, 4), dtype=torch.int32), -1)
        with self.assertRaises(ValueError):
            mt.build_hierarchy(torch.tensor([[0, 1 << 21, 0, 0]], dtype=torch.int32))


if __name__ == "__main__":
    unittest.main()
