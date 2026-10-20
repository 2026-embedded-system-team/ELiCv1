"""Command-line frontend; explicit flags override JSON configuration."""
import argparse
import json
from pathlib import Path
import sys

from .api import discover_environment, run_benchmarks
from .config import ConfigError, normalize_config


def parser():
    root = argparse.ArgumentParser(description="Portable ELiCv1 exact-correctness and paired performance runner")
    sub = root.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Inspect environment; no packages are installed")
    doctor.add_argument("--repo-root")
    doctor.add_argument("--output", help="Optional diagnostic JSON file")
    run = sub.add_parser("run", help="Run correctness and paired latency suites")
    run.add_argument("--config", type=Path)
    run.add_argument("--repo-root")
    run.add_argument("--output-dir")
    run.add_argument("--device")
    run.add_argument("--suites", nargs="+")
    for name in ("points", "channels", "warmups", "repeats", "seed", "threads", "timeout"):
        run.add_argument("--" + name, type=int)
    run.add_argument("--posq", nargs="+", type=int)
    run.add_argument("--compiler")
    run.add_argument("--files", nargs="+")
    run.add_argument("--prequantized", choices=("auto", "yes", "no"))
    for name in ("synthetic", "correctness-only", "async-entropy"):
        run.add_argument("--" + name, action=argparse.BooleanOptionalAction, default=None)
    return root


def main(argv=None):
    arguments = parser().parse_args(argv)
    try:
        if arguments.command == "doctor":
            result = discover_environment(arguments.repo_root)
            rendered = json.dumps(result, ensure_ascii=False, indent=2)
            if arguments.output:
                path = Path(arguments.output).expanduser().resolve()
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(rendered + "\n", encoding="utf-8")
            print(rendered)
            return 0
        values = {}
        if arguments.config:
            path = arguments.config.expanduser().resolve()
            values = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(values, dict):
                raise ConfigError("Configuration must be a JSON object")
            # Resolve only supplied paths here. Full normalization happens
            # after overrides, so --prequantized and --synthetic still affect
            # config-provided paths and automatically discovered samples.
            def from_config(value):
                if not isinstance(value, str):
                    raise ConfigError("Paths in a JSON configuration must be strings")
                item = Path(value).expanduser()
                return str(item.resolve() if item.is_absolute() else (path.parent / item).resolve())
            for key in ("repo_root", "output_dir"):
                if values.get(key) is not None:
                    values[key] = from_config(values[key])
            if isinstance(values.get("files"), list):
                resolved = []
                for entry in values["files"]:
                    if isinstance(entry, str):
                        resolved.append(from_config(entry))
                    elif isinstance(entry, dict) and isinstance(entry.get("path"), str):
                        resolved.append({**entry, "path": from_config(entry["path"])})
                    else:
                        resolved.append(entry)
                values["files"] = resolved
        values.update({key: value for key, value in vars(arguments).items()
                       if key not in ("command", "config") and value is not None})
        result = run_benchmarks(values)
        print(json.dumps(dict(status=result["summary"]["status"], summary=result["summary"],
                              output_dir=result["config"]["output_dir"], backend=result["backend"]), ensure_ascii=False, indent=2))
        return result["summary"]["exit_code"]
    except (ConfigError, OSError, ValueError) as error:
        print(json.dumps(dict(status="configuration_error", exit_code=2, reason=str(error)), ensure_ascii=False), file=sys.stderr)
        return 2
