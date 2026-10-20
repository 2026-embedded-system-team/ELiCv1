"""Isolated writer experiment; add --benchmark only when the CPU is idle."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.benchmark_arithmetic import extract_reference, geometry_histogram_samples

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    compiler = shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        raise RuntimeError("g++ or clang++ is required")
    subprocess.run([sys.executable, str(HERE / "prepare.py")], check=True)
    build = HERE / ".build"
    build.mkdir(exist_ok=True)
    (build / "reference_core.h").write_text(extract_reference(), encoding="utf-8")
    flags = ["-O3", "-std=c++17", "-DNDEBUG"]
    commands = []
    executables = []
    for source, label in [(ROOT / "benchmarks/arithmetic_test.cpp", "compatibility"), (HERE / "writer_test.cpp", "writer")]:
        executable = build / (label + (".exe" if os.name == "nt" else ""))
        command = [compiler, *flags, f"-I{HERE}", f"-I{build}", str(source), "-o", str(executable)]
        subprocess.run(command, check=True, capture_output=True, text=True)
        commands.append(command)
        executables.append(executable)
    results = {}
    for executable in executables:
        completed = subprocess.run([str(executable)], check=True, capture_output=True, text=True)
        results[executable.stem] = json.loads(completed.stdout)
    paths, inputs = geometry_histogram_samples(build)
    if args.benchmark:
        completed = subprocess.run([str(executables[1]), "--benchmark", *paths], check=True, capture_output=True, text=True)
        results["timings"] = json.loads(completed.stdout)
    files = [HERE / name for name in ["before_arithmetic_core.h", "arithmetic_core.h", "writer_test.cpp", "run.py", "prepare.py"]]
    files += [ROOT / "benchmarks/arithmetic_test.cpp", ROOT / "benchmarks/benchmark_arithmetic.py"]
    results.update(created_utc=datetime.now(timezone.utc).isoformat(),
        compiler=compiler, compiler_flags=flags, compile_commands=commands,
        compiler_version=subprocess.run([compiler, "--version"], check=True, capture_output=True, text=True).stdout.splitlines()[0],
        environment=dict(python=sys.version, platform=platform.platform(), processor=os.environ.get("PROCESSOR_IDENTIFIER") or platform.processor()),
        source_hashes={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        geometry_histogram_inputs=inputs,
        timing_config=dict(warmups=3, repeats=15, order="alternating pairs", aggregation="median", clock="std::chrono::steady_clock", unit="milliseconds",
            includes=["encoder reset", "encode_chunk", "finalize"], excludes=["CDF generation", "input I/O", "Torch bindings", "neural network", "GPU", "output string copy"]),
        comparison="current production snapshot versus stack-batched writer; frozen original used for compatibility only")
    output = (args.output or HERE / ("timing.json" if args.benchmark else "correctness.json")).resolve()
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "compatibility_cases": results["compatibility"]["cases"],
                      "writer_append_checks": results["writer"]["writer_append_checks"], "timed": args.benchmark}))

if __name__ == "__main__":
    main()
