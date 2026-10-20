"""Distribution checks that run before any NumPy/PyTorch installation."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile


SPEC = importlib.util.spec_from_file_location("elic_bundle", Path(__file__).resolve().parents[1] / "bundle.py")
BUNDLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUNDLE)


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="elic-package-test-")
        self.root = Path(self.temporary.name)
        self.repo, self.tool = self.root / "source repo", self.root / "tool source"
        for name in BUNDLE.REQUIRED_FILES:
            self.write(self.repo / name, "{}" if name.endswith(".json") else "// fixture\n")
        self.write(self.repo / "LICENSE", "BSD fixture license\n")
        for name in BUNDLE.SAMPLE_FILES:
            self.write(self.repo / name, "ply\nformat ascii 1.0\nend_header\n")
        self.write(self.repo / "model/Ford/ELiCv1_K5.pt", "model fixture")
        self.write(self.repo / "model/Ford/Centers_5.npy", "centers fixture")
        self.write(self.repo / "reports/private.json", "must not distribute")
        self.write(self.repo / ".git/config", "must not distribute")
        self.write(self.repo / "csrc/generated.dll", "must not distribute")
        self.write(self.tool / "run.py", "from pathlib import Path\nprint((Path(__file__).resolve().parent.parent/'runtime/src/op.py').is_file())\n")
        self.write(self.tool / "elic_bench/native_bridge.cpp", "// native source\n")
        self.write(self.tool / "elic_bench/__init__.py", "")
        self.write(self.tool / "results/previous.json", "must not distribute")
        self.write(self.tool / "dist/previous.zip", "must not distribute")
        self.archive = self.root / "output package.zip"

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")

    def create(self, **options):
        return BUNDLE.create_bundle(self.repo, self.archive, tool_root=self.tool, **options)

    def test_default_has_samples_native_sources_and_licenses_without_generated_files(self):
        result = self.create()
        self.assertTrue(result["samples_included"])
        self.assertFalse(result["models_included"])
        checked = BUNDLE.verify_bundle(self.archive)
        self.assertEqual(result["files"], checked["files"])
        with zipfile.ZipFile(self.archive) as archive:
            names = archive.namelist()
            self.assertIn(f"{BUNDLE.PACKAGE}/portable_bench/elic_bench/native_bridge.cpp", names)
            self.assertIn(f"{BUNDLE.PACKAGE}/runtime/LICENSE", names)
            for sample in BUNDLE.SAMPLE_FILES:
                self.assertIn(f"{BUNDLE.PACKAGE}/runtime/{sample}", names)
            self.assertFalse(any("reports/" in name or ".git/" in name or "generated.dll" in name or "previous" in name for name in names))
            manifest = json.loads(archive.read(f"{BUNDLE.PACKAGE}/manifest.json"))
            self.assertTrue(all(len(entry["sha256"]) == 64 for entry in manifest["files"]))

    def test_options_include_models_and_omit_samples(self):
        result = self.create(include_models=True, no_samples=True)
        self.assertFalse(result["samples_included"])
        with zipfile.ZipFile(self.archive) as archive:
            names = archive.namelist()
            self.assertIn(f"{BUNDLE.PACKAGE}/runtime/model/Ford/ELiCv1_K5.pt", names)
            self.assertIn(f"{BUNDLE.PACKAGE}/runtime/model/Ford/Centers_5.npy", names)
            self.assertFalse(any("/data/" in name for name in names))

    def test_extracted_package_can_be_moved_and_used_from_unrelated_cwd(self):
        self.create(no_samples=True)
        extracted = self.root / "first location"
        with zipfile.ZipFile(self.archive) as archive:
            archive.extractall(extracted)
        moved = self.root / "relocated package"
        (extracted / BUNDLE.PACKAGE).rename(moved)
        completed = subprocess.run([sys.executable, str(moved / "portable_bench/run.py")],
                                   cwd=self.root, text=True, capture_output=True, check=True)
        self.assertEqual(completed.stdout.strip(), "True")

    def test_changed_payload_is_detected(self):
        self.create(no_samples=True)
        damaged = self.root / "damaged.zip"
        with zipfile.ZipFile(self.archive) as incoming, zipfile.ZipFile(damaged, "w") as outgoing:
            for info in incoming.infolist():
                payload = incoming.read(info.filename)
                if info.filename.endswith("runtime/src/op.py"):
                    payload += b"modified"
                outgoing.writestr(info.filename, payload)
        with self.assertRaisesRegex(ValueError, "integrity"):
            BUNDLE.verify_bundle(damaged)

    def test_extra_unsafe_member_is_detected(self):
        self.create(no_samples=True)
        with zipfile.ZipFile(self.archive, "a") as archive:
            archive.writestr("../escape.txt", "bad")
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            BUNDLE.verify_bundle(self.archive)

    def test_malformed_manifest_is_explicit_error(self):
        with zipfile.ZipFile(self.archive, "w") as archive:
            archive.writestr(f"{BUNDLE.PACKAGE}/manifest.json", "[]")
        with self.assertRaisesRegex(ValueError, "manifest"):
            BUNDLE.verify_bundle(self.archive)

    def test_missing_source_or_sample_is_explicit_error(self):
        (self.repo / BUNDLE.SAMPLE_FILES[0]).unlink()
        with self.assertRaisesRegex(ValueError, "Sample missing"):
            self.create()
        self.create(no_samples=True)
        (self.repo / "src/op.py").unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            self.create(no_samples=True)


if __name__ == "__main__":
    unittest.main()
