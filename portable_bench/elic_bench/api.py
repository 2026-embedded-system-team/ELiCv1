"""Public Python API; optional numerical dependencies are loaded lazily."""
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

from . import __version__
from .config import default_repo_root, normalize_config
from .results import summarize, write_outputs


def discover_environment(repo_root=None):
    result = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                  machine=platform.machine(), logical_cpus=os.cpu_count(),
                  processor=os.environ.get("PROCESSOR_IDENTIFIER") or platform.processor(),
                  repo_root=str(Path(repo_root or default_repo_root()).resolve()), packages={},
                  tools={name: shutil.which(name) for name in ("g++", "clang++", "cl", "nvcc", "nvidia-smi")})
    for name in ("numpy", "torch", "pytest", "pandas", "ninja", "open3d", "tqdm", "torchsparse"):
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = None
    try:
        import torch
        available = torch.cuda.is_available()
        result["cuda"] = dict(available=available, torch_cuda_version=torch.version.cuda,
                              devices=[dict(index=i, name=torch.cuda.get_device_name(i),
                                            total_memory_bytes=torch.cuda.get_device_properties(i).total_memory,
                                            capability=list(torch.cuda.get_device_capability(i)))
                                       for i in range(torch.cuda.device_count())] if available else [])
    except Exception as error:
        result["cuda"] = dict(available=False, devices=[], reason=f"{type(error).__name__}: {error}")
    repo = Path(result["repo_root"])
    result["runtime_found"] = all((repo / path).is_file() for path in ("src/morton.py", "src/op.py", "csrc/arithmetic_core.h", "benchmarks/reference/manifest.json"))
    result["capabilities"] = dict(core_numerical=result["packages"]["torch"] is not None and result["packages"]["numpy"] is not None,
                                  native_compiler=any(result["tools"][x] for x in ("g++", "clang++", "cl")),
                                  real_cuda=result["cuda"]["available"], cuda_interface_simulation=True,
                                  neural_dependencies_found=all(result["packages"][x] is not None for x in ("torchsparse", "open3d", "ninja", "pandas", "tqdm")))
    return result


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sources(repo):
    paths = sorted((repo / "src").glob("*.py")) + sorted((repo / "csrc").glob("*.h")) + sorted((repo / "csrc").glob("*.cpp"))
    paths += [repo / "model.py", repo / "compress.py", repo / "decompress.py"]
    return {str(path.relative_to(repo)).replace("\\", "/"): _hash(path) for path in paths if path.is_file()}


def _tool_sources():
    folder = Path(__file__).resolve().parent
    paths = sorted(folder.glob("*.py")) + sorted(folder.glob("*.cpp")) + [folder.parent / "run.py"]
    return {str(path.relative_to(folder.parent)).replace("\\", "/"): _hash(path) for path in paths}


def _baseline_integrity(repo):
    root = repo / "benchmarks/reference"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or _hash(path) != expected:
            raise ValueError(f"Frozen baseline integrity mismatch: {relative}")
    return dict(commit=manifest.get("commit", manifest.get("baseline_commit")),
                manifest_sha256=_hash(root / "manifest.json"), verified_files=len(manifest["files"]))


def run_benchmarks(config=None):
    """Run selected suites and persist result.json/results.csv/junit.xml.

    ``config`` has the same JSON keys as run.py --config. Relative paths use
    the current working directory; CLI config-file paths use its parent.
    The return value is the complete schema-versioned JSON result dictionary.
    """
    normalized = normalize_config(config)
    output = Path(normalized["output_dir"])
    if (output / "result.json").exists():
        from .config import ConfigError
        raise ConfigError(f"An existing result would be overwritten: {output}; select a fresh output_dir")
    output.mkdir(parents=True, exist_ok=True)
    environment = discover_environment(normalized["repo_root"])
    report = dict(schema_version="1.0", tool=dict(name="elic-portable-bench", version=__version__),
                  created_utc=datetime.now(timezone.utc).isoformat(), config=normalized,
                  environment=environment, backend=None, provenance={}, cases=[])
    started = time.perf_counter()
    context = None
    try:
        if not __debug__:
            raise RuntimeError("Python optimization removes required correctness assertions. Run without -O/-OO and unset PYTHONOPTIMIZE.")
        if sys.byteorder != "little":
            raise RuntimeError("The frozen binary/CDF interfaces use native byte order; this portable distribution currently supports little-endian hosts only.")
        if not environment["runtime_found"]:
            raise RuntimeError("ELiCv1 runtime is incomplete; use --repo-root or distribute a bundle created with bundle.py")
        if not environment["capabilities"]["core_numerical"]:
            raise ModuleNotFoundError("NumPy and PyTorch are required to run numerical suites; doctor and bundle commands work without them")
        import torch
        from .backend import Backend
        from .context import Context
        backend = Backend(normalized["device"], torch)
        torch.set_num_threads(normalized["threads"])
        torch.manual_seed(normalized["seed"])
        if backend.device.type == "cuda":
            torch.cuda.manual_seed_all(normalized["seed"])
        context = Context(normalized, backend)
        report["backend"] = backend.describe()
        report["backend"]["timing_enabled"] = not (normalized["correctness_only"] or backend.simulated)
        report["provenance"] = dict(baseline=_baseline_integrity(context.repo_root),
                                     candidate_source_sha256=_sources(context.repo_root),
                                     tool_source_sha256=_tool_sources(),
                                     input_sha256={item["path"]: _hash(Path(item["path"])) for item in normalized["files"]},
                                     baseline_kind="frozen original ELiCv1 algorithms with interface-only adaptations")
        from .suites import run_modules, run_io, run_geometry
        with ExitStack() as stack:
            coders = None
            native_error = None
            if any(suite in normalized["suites"] for suite in ("arithmetic", "geometry")):
                try:
                    from .arithmetic import build_coders
                    coders = stack.enter_context(build_coders(context.repo_root, output / "build/native", normalized["compiler"]))
                    report["native_build"] = coders[2]
                except Exception as error:
                    from .native import NativeUnavailable
                    native_error = ("unavailable" if isinstance(error, NativeUnavailable) else "error", f"{type(error).__name__}: {error}")
            # The module suite verifies feature autograd as part of the public
            # API. Integer/no-grad primitives keep their own production guards.
            with torch.inference_mode(False), torch.enable_grad():
                for suite in normalized["suites"]:
                    try:
                        if suite in ("arithmetic", "geometry") and native_error:
                            context.add_result(f"{suite}.capability", suite, native_error[0], native_error[1])
                        elif suite == "modules":
                            list(run_modules(context))
                        elif suite == "io":
                            list(run_io(context))
                        elif suite == "geometry":
                            list(run_geometry(context, coders[0], coders[1]))
                        elif suite == "arithmetic":
                            from .arithmetic import run_arithmetic
                            list(run_arithmetic(context, coders[0], coders[1]))
                        else:
                            from .external import run_binding, run_pipeline
                            (run_binding if suite == "binding" else run_pipeline)(context)
                    except Exception as error:
                        context.add_result(f"{suite}.runner_error", suite, "error", f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        assert _sources(context.repo_root) == report["provenance"]["candidate_source_sha256"], "Source files changed while measuring; rerun with a stable checkout"
        assert _tool_sources() == report["provenance"]["tool_source_sha256"], "Benchmark harness changed while measuring; rerun after edits finish"
    except KeyboardInterrupt:
        if context:
            context.add_result("run.interrupted", "environment", "error", "Interrupted by user")
        else:
            report["cases"].append(dict(case_id="run.interrupted", suite="environment", status="error", reason="Interrupted by user"))
        report["interrupted"] = True
    except Exception as error:
        status = "unavailable" if context is None else "error"
        case = dict(case_id="environment.preflight" if context is None else "environment.integrity", suite="environment",
                    status=status, reason=f"{type(error).__name__}: {error}", correctness={"check_passed": None}, timing=None,
                    traceback=traceback.format_exc())
        if context:
            context.records.append(case)
        else:
            report["cases"].append(case)
    if context:
        report["cases"] = context.records
    report["summary"] = summarize(report["cases"])
    if report.get("interrupted"):
        report["summary"].update(exit_code=130, status="interrupted", coverage_complete=False)
    report["elapsed_seconds"] = time.perf_counter()-started
    write_outputs(report, output)
    return report
