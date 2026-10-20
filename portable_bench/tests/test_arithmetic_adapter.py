"""Native integration checks; skip explicitly if no C++ toolchain is installed."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from elic_bench.arithmetic import build_coders, run_arithmetic, validate_cdf
from elic_bench.config import default_repo_root
from elic_bench.native import NativeBuildError, NativeUnavailable, _compiler_command


REPO = default_repo_root()


def fixture(n=511):
    rng = np.random.default_rng(20261020)
    boundaries = np.sort(rng.choice(np.arange(1, 65535), 15, replace=False))
    row = np.concatenate(([0], boundaries, [0])).astype(np.uint16).view(np.int16)
    cdf = np.tile(row, (n, 1))
    return cdf, rng.integers(0, 16, n, dtype=np.int16)


class TestArithmeticAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="elic-adapter-test-")
        cls.context = build_coders(REPO, Path(cls.temporary.name))
        try:
            cls.reference, cls.candidate, cls.info = cls.context.__enter__()
        except NativeUnavailable as error:
            cls.temporary.cleanup()
            raise unittest.SkipTest(str(error))
        except BaseException:
            cls.temporary.cleanup()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.context.__exit__(None, None, None)
        cls.temporary.cleanup()

    def test_byte_compatibility_cross_decode_and_streaming(self):
        cdf, symbols = fixture()
        streams = []
        for module in (self.reference, self.candidate):
            with module.ArithmeticEncoder16() as encoder:
                for start, end in ((0, 0), (0, 3), (3, 67), (67, len(symbols))):
                    encoder.encode_chunk(cdf[start:end], symbols[start:end])
                self.assertTrue(0 <= encoder.buffered_bitcount() < 8)
                self.assertIsInstance(encoder.get_buffer_bytes(), bytes)
                streams.append(encoder.finish())
                self.assertEqual(encoder.get_buffer_bytes(), b"")
                self.assertEqual(encoder.buffered_bitcount(), 0)
            self.assertEqual(module.encode_cdf_16(cdf, symbols), streams[-1])
        self.assertEqual(streams[0], streams[1])
        for module in (self.reference, self.candidate):
            for stream in streams:
                with module.ArithmeticDecoder16() as decoder:
                    decoder.reset(stream)
                    chunks = [decoder.decode_chunk(cdf[:19]), decoder.decode_chunk(cdf[19:])]
                    np.testing.assert_array_equal(np.concatenate(chunks), symbols)
                    self.assertGreaterEqual(decoder.bytes_consumed(), 0)
                    self.assertGreaterEqual(decoder.bytes_remaining(), 0)
                    decoder.reset(stream)
                    np.testing.assert_array_equal(decoder.decode_chunk(cdf), symbols)
                np.testing.assert_array_equal(module.decode_cdf_16(cdf, stream), symbols)

    def test_finish_and_reset_options_match_reference(self):
        cdf, symbols = fixture(29)
        for reset_after in (False, True):
            for clear_output_after in (False, True):
                outputs = []
                for module in (self.reference, self.candidate):
                    with module.ArithmeticEncoder16() as encoder:
                        encoder.encode_chunk(cdf, symbols)
                        first = encoder.finish(reset_after, clear_output_after)
                        buffer = encoder.get_buffer_bytes()
                        encoder.reset(reset_output=False)
                        encoder.encode_chunk(cdf, symbols)
                        second = encoder.finish()
                        encoder.reset()
                        encoder.encode_chunk(cdf, symbols)
                        third = encoder.finish()
                        outputs.append((first, buffer, second, third))
                self.assertEqual(outputs[0], outputs[1])

    def test_empty_and_small_tail_streams(self):
        for n in (0, 1, 2, 3, 4, 7, 9, 33):
            cdf, symbols = fixture(n)
            old = self.reference.encode_cdf_16(cdf, symbols)
            new = self.candidate.encode_cdf_16(cdf, symbols)
            self.assertEqual(old, new)
            np.testing.assert_array_equal(self.reference.decode_cdf_16(cdf, new), symbols)
            np.testing.assert_array_equal(self.candidate.decode_cdf_16(cdf, old), symbols)

    def test_input_contract_and_symbols_validated_before_native_call(self):
        cdf, symbols = fixture(3)
        for module in (self.reference, self.candidate):
            with module.ArithmeticEncoder16() as encoder:
                for wrong in (symbols.astype(np.int32), symbols.tolist()):
                    with self.assertRaises(TypeError):
                        encoder.encode_chunk(cdf, wrong)
                for value in (-1, 16):
                    bad = symbols.copy()
                    bad[0] = value
                    with self.assertRaises(ValueError):
                        encoder.encode_chunk(cdf, bad)
                with self.assertRaises(ValueError):
                    encoder.encode_chunk(cdf, symbols[:1])
                with self.assertRaises(ValueError):
                    encoder.encode_chunk(cdf[:, ::-1], symbols)
                with self.assertRaises(ValueError):
                    encoder.encode_chunk(cdf[:, :16].copy(), symbols)
                bad_cdf = cdf.copy()
                bad_cdf[0, 2] = bad_cdf[0, 1]
                with self.assertRaises(ValueError):
                    encoder.encode_chunk(bad_cdf, symbols)
                # Rejected calls leave the encoder untouched.
                encoder.encode_chunk(cdf, symbols)
                self.assertEqual(encoder.finish(), module.encode_cdf_16(cdf, symbols))
            with module.ArithmeticDecoder16() as decoder:
                with self.assertRaises(TypeError):
                    decoder.reset(bytearray(b"a"))
                with self.assertRaises(TypeError):
                    decoder.decode_chunk(cdf.astype(np.uint16))

    def test_explicit_close_is_idempotent(self):
        for module in (self.reference, self.candidate):
            encoder = module.ArithmeticEncoder16()
            encoder.close()
            encoder.close()
            with self.assertRaises(RuntimeError):
                encoder.finish()
            decoder = module.ArithmeticDecoder16()
            decoder.close()
            with self.assertRaises(RuntimeError):
                decoder.reset(b"")

    def test_unchecked_values_path_has_same_valid_output(self):
        cdf, symbols = fixture()
        validate_cdf(cdf)
        for module in (self.reference, self.candidate):
            with module.ArithmeticEncoder16(validate=False) as encoder:
                encoder.encode_chunk(cdf, symbols)
                stream = encoder.finish()
            with module.ArithmeticDecoder16(validate=False) as decoder:
                decoder.reset(stream)
                np.testing.assert_array_equal(decoder.decode_chunk(cdf), symbols)

    def test_build_metadata(self):
        self.assertGreater(self.info["compile_seconds"], 0)
        self.assertTrue(self.info["compiler_command"])
        self.assertTrue(self.info["compiler_version"])
        for field in ("reference_sha256", "reference_adapter_sha256", "core_sha256", "bridge_sha256"):
            self.assertEqual(len(self.info[field]), 64)

    def test_suite_records_all_distributions_and_tiny_inputs(self):
        from elic_bench.context import Context
        backend = SimpleNamespace(device=SimpleNamespace(type="cpu"), simulated=False,
                                  synchronize=lambda: None)
        for points in (1, 64):
            config = dict(repo_root=str(REPO), output_dir=self.temporary.name,
                          points=points, seed=11, correctness_only=True)
            ctx = Context(config, backend)
            records = list(run_arithmetic(ctx, self.reference, self.candidate))
            self.assertEqual(len(records), 11)
            for record in records:
                self.assertEqual(record["status"], "passed", record.get("traceback"))
                self.assertIsNone(record["timing"])


class TestNativeBuildLifecycle(unittest.TestCase):
    def test_no_toolchain_is_explicitly_unavailable(self):
        with self.assertRaises(NativeUnavailable):
            _compiler_command("elic-no-such-compiler-20261020")

    def test_cxx_environment_is_respected(self):
        with mock.patch.dict("os.environ", {"CXX": "elic-no-such-compiler-20261020"}):
            with self.assertRaises(NativeUnavailable):
                _compiler_command(None)

    def test_dll_closed_and_temporary_build_removed(self):
        with tempfile.TemporaryDirectory(prefix="elic-unload-test-") as temporary:
            build = Path(temporary)
            try:
                with build_coders(REPO, build) as (_, module, _):
                    encoder = module.ArithmeticEncoder16()
                    decoder = module.ArithmeticDecoder16()
                    self.assertTrue(list(build.iterdir()))
            except NativeUnavailable as error:
                self.skipTest(str(error))
            self.assertEqual(list(build.iterdir()), [])
            with self.assertRaises(RuntimeError):
                encoder.finish()
            with self.assertRaises(RuntimeError):
                decoder.reset(b"")
            with self.assertRaises(RuntimeError):
                module.ArithmeticEncoder16()

    def test_missing_sources_are_build_failure(self):
        with tempfile.TemporaryDirectory(prefix="elic-source-test-") as temporary:
            with self.assertRaises(NativeBuildError):
                with build_coders(Path(temporary), Path(temporary) / "output"):
                    self.fail("Missing source must not produce a coder")


if __name__ == "__main__":
    unittest.main()
