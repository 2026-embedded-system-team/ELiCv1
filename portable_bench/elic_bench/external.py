"""Optional production integrations staged entirely in the run directory."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def run_process(command, cwd, log_path, timeout, env=None):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(list(map(str, command)), cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT, **options)
        try:
            code = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
            else:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait()
            raise
    return code, time.perf_counter()-started


def _stage(ctx, name, paths):
    stage = ctx.output_dir / "work" / (name + "_runtime")
    stage.mkdir(parents=True, exist_ok=True)
    for relative in paths:
        source = ctx.repo_root / relative
        destination = stage / relative
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__", ".build", "*.pyc"))
        elif source.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        else:
            raise FileNotFoundError(f"Required integration source missing: {source}")
    (stage / "reports").mkdir(exist_ok=True)
    return stage


def run_binding(ctx):
    if ctx.simulated:
        return ctx.add_result("binding.capability", "binding", "unavailable", "Production Torch extension compilation is not simulated; run with device cpu or a real CUDA device.")
    stage = _stage(ctx, "binding", ["csrc", "benchmarks/check_torch_binding.py"])
    target = stage / "binding_result.json"
    log = ctx.output_dir / "logs" / "binding.log"
    command = [sys.executable, stage / "benchmarks/check_torch_binding.py", "--output", target]
    code, elapsed = run_process(command, stage, log, ctx.config["timeout"])
    data = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
    details = dict(command=list(map(str, command)), log=str(log), integration_result=data,
                   process_wall_seconds=elapsed, uses_cuda=False, timing_scope="correctness_only")
    if code == 0 and data.get("status") == "passed":
        case = ctx.add_result("binding.production_api", "binding", "passed", details=details)
        case["correctness"] = {"check_passed": True, "metrics": {"checks": data.get("checks", [])}}
        return case
    reason = data.get("reason", f"Production binding process exited {code}; see {log}")
    lowered = reason.lower()
    missing_toolchain = any(marker in lowered for marker in (
        "visual studio c++ compiler was not found", "microsoft visual c++ 14.0 or greater is required",
        "windows sdk", "cannot open include file: 'assert.h'", "cannot open include file: 'windows.h'",
        "error: microsoft visual c++", "no module named 'setuptools'"))
    # The legacy helper labels every build exception unavailable; distinguish
    # missing SDKs from syntax/ABI regressions in the production source.
    status = "failed" if "smoke test failed" in lowered else "unavailable" if missing_toolchain else "error"
    return ctx.add_result("binding.production_api", "binding", status, reason, details=details)


def run_pipeline(ctx):
    if ctx.simulated or ctx.device.type != "cuda":
        return ctx.add_result("pipeline.capability", "pipeline", "unavailable", "The trained TorchSparse neural pipeline requires a real CUDA device; CPU/interface simulation covers only the selected core suites.")
    missing = [name for name in ("torchsparse", "open3d", "ninja", "pandas", "tqdm") if importlib.util.find_spec(name) is None]
    if missing:
        return ctx.add_result("pipeline.capability", "pipeline", "unavailable", "Missing neural pipeline dependencies: " + ", ".join(missing))
    if ctx.config["synthetic"]:
        return ctx.add_result("pipeline.capability", "pipeline", "unavailable", "Pipeline integration uses the example Ford/SemanticKITTI PLY inputs and checkpoints; disable --synthetic.")
    default_files = sorted(str(p.resolve()) for p in (ctx.repo_root / "data").glob("*Example/*.ply"))
    if any(not list((ctx.repo_root / "data" / f"{name}Example").glob("*.ply")) for name in ("Ford", "SemanticKITTI")):
        return ctx.add_result("pipeline.capability", "pipeline", "unavailable", "Full pipeline requires both example datasets; create a bundle without --no-samples or supply a complete --repo-root.")
    if sorted(f["path"] for f in ctx.config["files"]) != default_files:
        return ctx.add_result("pipeline.capability", "pipeline", "unavailable", "The legacy full-pipeline acceptance interface supports the bundled example datasets only. Custom input files are supported by modules/io/geometry.")
    required = [f"model/{dataset}/ELiCv1_K5.pt" for dataset in ("Ford", "SemanticKITTI")]
    missing_files = [path for path in required if not (ctx.repo_root / path).is_file()]
    if missing_files:
        return ctx.add_result("pipeline.capability", "pipeline", "unavailable", "Checkpoints absent; bundle with --include-models or supply --repo-root: " + ", ".join(missing_files))
    stage = _stage(ctx, "pipeline", ["src", "csrc", "model", "data/FordExample", "data/SemanticKITTIExample",
                                   "model.py", "compress.py", "decompress.py", "benchmarks/reference", "benchmarks/compare_pipeline.py"])
    target = stage / "pipeline_result.json"
    log = ctx.output_dir / "logs" / "pipeline.log"
    command = [sys.executable, stage / "benchmarks/compare_pipeline.py", "--posQ", *ctx.config["posq"],
               "--repeats", 1 if ctx.config["correctness_only"] else ctx.config["repeats"], "--output", target]
    if ctx.config["async_entropy"]:
        command.append("--async_entropy")
    environment = os.environ.copy()
    # Remap the selected physical GPU to ordinal zero for all legacy drivers.
    visible = environment.get("CUDA_VISIBLE_DEVICES")
    environment["CUDA_VISIBLE_DEVICES"] = visible.split(",")[ctx.device.index].strip() if visible else str(ctx.device.index)
    environment["TORCH_EXTENSIONS_DIR"] = str(stage / ".torch_extensions")
    code, elapsed = run_process(command, stage, log, ctx.config["timeout"], environment)
    data = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
    if ctx.config["correctness_only"]:
        for row in data.get("rows", []):
            for key in ("enc", "dec", "process", "samples_seconds"):
                row.pop(key, None)
        data.pop("timing_note", None)
        data["performance_measurements_suppressed"] = True
        if target.exists():
            target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    details = dict(command=list(map(str, command)), log=str(log), integration_result=data,
                   process_wall_seconds=elapsed, timing_scope="integration statistics in integration_result.rows",
                   uses_cuda=True, external_timer="legacy baseline CUDA events; candidate synchronized host wall clock; process wall time recorded separately")
    if code == 0 and data.get("status") == "passed":
        case = ctx.add_result("pipeline.neural_cross_decode", "pipeline", "passed", details=details)
        case.update(cuda_execution_validated=True, correctness={"check_passed": True, "metrics": {"byte_identical": True, "exact_quantized_geometry": True, "bidirectional_cross_decode": True}})
        return case
    return ctx.add_result("pipeline.neural_cross_decode", "pipeline", "unavailable" if code == 2 else "error",
                          data.get("reason", f"Neural acceptance process exited {code}; see {log}"), details=details)
