"""Create and verify a relocatable ELiC benchmark ZIP using only the standard library."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
import zipfile


PACKAGE = "elic-portable-bench"
TOOL_ROOT = Path(__file__).resolve().parent
SOURCE_EXTENSIONS = {".py", ".md", ".json", ".txt", ".toml", ".sh", ".ps1", ".cpp", ".h"}
BENCHMARK_FILES = (
    "arithmetic_bridge.py", "arithmetic_ctypes.cpp", "arithmetic_test.cpp",
    "benchmark_arithmetic.py", "bench_bitreader_batch.py", "bitreader_batch_test.cpp",
    "bench_boe.py", "bench_io.py", "bench_memory_io.py", "bench_modules.py",
    "bench_morton.py", "check_torch_binding.py", "compare_pipeline.py",
)
REQUIRED_FILES = (
    "src/op.py", "src/morton.py", "src/io.py", "src/runtime.py", "src/boe.py",
    "csrc/arithmetic_core.h", "csrc/arithmetic_coding.cpp",
    "benchmarks/reference/manifest.json", "benchmarks/reference/csrc/arithmetic_coding.cpp",
)
SAMPLE_FILES = tuple(
    f"data/{dataset}Example/sample_{number:04d}.ply"
    for dataset in ("Ford", "SemanticKITTI") for number in (1, 2)
)


def default_repo_root() -> Path:
    packaged = TOOL_ROOT.parent / "runtime"
    return packaged if packaged.is_dir() else TOOL_ROOT.parent


def _safe_source(path: Path, root: Path) -> Path:
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Source path escapes its source root: {path}")
    if path.is_symlink():
        raise ValueError(f"Symbolic links are not distributable source files: {path}")
    return resolved


def collect_files(repo_root: Path, tool_root: Path = TOOL_ROOT, *, include_models: bool = False,
                  no_samples: bool = False) -> list[tuple[Path, str, str]]:
    """Return (source, package-relative POSIX path, category), using source allowlists."""
    repo_root, tool_root = repo_root.resolve(), tool_root.resolve()
    missing = [name for name in REQUIRED_FILES if not (repo_root / name).is_file()]
    if missing:
        raise ValueError("Not a complete ELiC source root; missing: " + ", ".join(missing))
    if not any((repo_root / name).is_file() for name in ("LICENSE", "LICENSE.md")):
        raise ValueError("ELiC license file is required for redistribution")
    found: dict[str, tuple[Path, str, str]] = {}

    def add(path: Path, root: Path, prefix: str, category: str) -> None:
        source = _safe_source(path, root)
        relative = path.relative_to(root).as_posix()
        target = f"{prefix}/{relative}"
        found[target] = (source, target, category)

    # Only tool source directories are visited; results, caches, environments,
    # archives and build products cannot enter through a recursive wildcard.
    for directory in (tool_root, tool_root / "elic_bench", tool_root / "docs", tool_root / "configs", tool_root / "tests"):
        if not directory.is_dir():
            continue
        paths = directory.iterdir() if directory == tool_root else directory.rglob("*")
        for path in paths:
            if path.is_file() and path.suffix.lower() in SOURCE_EXTENSIONS:
                if any(part.startswith(".") or part == "__pycache__" for part in path.relative_to(tool_root).parts):
                    continue
                add(path, tool_root, "portable_bench", "tool")
    for name in (".gitignore", "Dockerfile"):
        if (tool_root / name).is_file():
            add(tool_root / name, tool_root, "portable_bench", "tool")
    for directory, extensions in (("src", {".py"}), ("csrc", {".h", ".cpp", ".cu"}),
                                  ("tests", {".py"})):
        for path in sorted((repo_root / directory).glob("*")):
            if path.is_file() and path.suffix in extensions:
                add(path, repo_root, "runtime", "runtime")
    for name in ("model.py", "compress.py", "decompress.py", "BoE_cluster.py", "eval.py",
                 "README.md", "LICENSE", "LICENSE.md"):
        path = repo_root / name
        if path.is_file():
            add(path, repo_root, "runtime", "license" if name.startswith("LICENSE") else "runtime")
    for name in BENCHMARK_FILES:
        path = repo_root / "benchmarks" / name
        if path.is_file():
            add(path, repo_root, "runtime", "runtime")
    for path in sorted((repo_root / "benchmarks/reference").rglob("*")):
        if path.is_file() and path.suffix in {".py", ".cpp", ".h", ".json"}:
            if "__pycache__" not in path.parts:
                add(path, repo_root, "runtime", "baseline")
    if not no_samples:
        for name in SAMPLE_FILES:
            if not (repo_root / name).is_file():
                raise ValueError(f"Sample missing: {name}; use --no-samples to omit samples")
            add(repo_root / name, repo_root, "runtime", "sample")
    if include_models:
        candidates = sorted((repo_root / "model").rglob("*"))
        weights = [path for path in candidates if path.is_file() and path.suffix in {".pt", ".pth", ".npy", ".json"}]
        if not weights or not any(path.suffix in {".pt", ".pth"} for path in weights):
            raise ValueError("--include-models requested but no model checkpoints were found")
        for path in weights:
            add(path, repo_root, "runtime", "model")
    if "portable_bench/run.py" not in found:
        raise ValueError("Tool run.py is missing; package preparation is incomplete")
    return [found[name] for name in sorted(found)]


def create_bundle(repo_root: Path, output: Path, *, tool_root: Path = TOOL_ROOT,
                  include_models: bool = False, no_samples: bool = False) -> dict:
    files = collect_files(repo_root, tool_root, include_models=include_models, no_samples=no_samples)
    output = output.resolve()
    if output.suffix.lower() != ".zip":
        raise ValueError("--output must end in .zip")
    if output in {source for source, _, _ in files}:
        raise ValueError("Output must not replace a distributed source file")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "package": PACKAGE,
        "samples_included": not no_samples,
        "models_included": include_models,
        "hash_algorithm": "sha256",
        "manifest_excludes_itself": True,
        "files": [],
    }
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(prefix="elic-bundle-", suffix=".zip", dir=output.parent, delete=False) as temporary:
            temporary_name = Path(temporary.name)
        with zipfile.ZipFile(temporary_name, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for source, relative, category in files:
                digest = hashlib.sha256()
                byte_count = 0
                with source.open("rb") as incoming, archive.open(f"{PACKAGE}/{relative}", "w", force_zip64=True) as outgoing:
                    while chunk := incoming.read(1024 * 1024):
                        outgoing.write(chunk)
                        digest.update(chunk)
                        byte_count += len(chunk)
                manifest["files"].append({"path": relative, "bytes": byte_count,
                                          "sha256": digest.hexdigest(), "category": category})
            archive.writestr(f"{PACKAGE}/manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        verify_bundle(temporary_name)
        os.replace(temporary_name, output)
        return {"output": str(output), "files": len(files), "bytes": output.stat().st_size,
                "payload_bytes": sum(item["bytes"] for item in manifest["files"]),
                "samples_included": not no_samples, "models_included": include_models}
    finally:
        if temporary_name is not None and temporary_name.exists():
            temporary_name.unlink()


def verify_bundle(archive_path: Path) -> dict:
    """Check archive member safety, exact membership, byte lengths and SHA-256, without extraction."""
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Archive contains duplicate members")
        for name in names:
            path = PurePosixPath(name)
            if not path.parts or path.is_absolute() or ".." in path.parts or "\\" in name or path.parts[0] != PACKAGE:
                raise ValueError(f"Unsafe archive member: {name}")
        manifest_name = f"{PACKAGE}/manifest.json"
        manifest = json.loads(archive.read(manifest_name))
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or manifest.get("hash_algorithm") != "sha256":
            raise ValueError("Unsupported bundle manifest")
        if not isinstance(manifest.get("files"), list):
            raise ValueError("Manifest files must be a list")
        expected = {manifest_name}
        for item in manifest["files"]:
            if (not isinstance(item, dict) or not isinstance(item.get("path"), str) or
                    type(item.get("bytes")) is not int or item["bytes"] < 0 or
                    not isinstance(item.get("sha256"), str) or len(item["sha256"]) != 64 or
                    any(character not in "0123456789abcdef" for character in item["sha256"])):
                raise ValueError("Malformed manifest file entry")
            name = f"{PACKAGE}/{item['path']}"
            if name in expected:
                raise ValueError(f"Duplicate manifest member: {name}")
            expected.add(name)
            digest, byte_count = hashlib.sha256(), 0
            with archive.open(name) as incoming:
                while chunk := incoming.read(1024 * 1024):
                    digest.update(chunk)
                    byte_count += len(chunk)
            if byte_count != item["bytes"] or digest.hexdigest() != item["sha256"]:
                raise ValueError(f"Bundle integrity check failed: {item['path']}")
        if set(names) != expected:
            raise ValueError("Archive members differ from manifest")
    return {"verified": True, "files": len(expected) - 1, "archive": str(Path(archive_path).resolve())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=default_repo_root(), help="ELiCv1 source root (default: sibling runtime, otherwise tool parent)")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output", type=Path, help="destination .zip; replaced atomically if it already exists")
    action.add_argument("--verify", type=Path, help="verify an existing ZIP without extraction")
    parser.add_argument("--include-models", action="store_true", help="include model checkpoints and BoE centers")
    parser.add_argument("--no-samples", action="store_true", help="omit the four example point clouds")
    args = parser.parse_args(argv)
    try:
        if args.verify:
            if args.include_models or args.no_samples:
                parser.error("--include-models and --no-samples are creation options")
            result = verify_bundle(args.verify)
        else:
            result = create_bundle(args.repo_root, args.output, include_models=args.include_models, no_samples=args.no_samples)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, zipfile.BadZipFile) as error:
        print(f"Bundle error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
