"""Strict, JSON-compatible configuration with explicit path resolution."""
from datetime import datetime, timezone
import glob
from os import PathLike
from pathlib import Path
import re
import uuid


class ConfigError(ValueError):
    pass


SUITES = ("modules", "io", "arithmetic", "geometry", "binding", "pipeline")
DEFAULTS = dict(device="auto", suites=list(SUITES[:4]), points=4096, channels=32,
                posq=[16], warmups=2, repeats=7, seed=11, threads=1,
                timeout=600, compiler=None, files=None, prequantized="auto",
                synthetic=False, correctness_only=False, async_entropy=False)


def default_repo_root():
    parent = Path(__file__).resolve().parents[2]
    return parent / "runtime" if (parent / "runtime").is_dir() else parent


def normalize_config(values=None, base_dir=None):
    if values is not None and not isinstance(values, dict):
        raise ConfigError("Configuration must be a dictionary/JSON object")
    values = dict(values or {})
    allowed = set(DEFAULTS) | {"repo_root", "output_dir"}
    unknown = set(values) - allowed
    if unknown:
        raise ConfigError("Unknown configuration keys: " + ", ".join(sorted(unknown)))
    base = Path(base_dir or Path.cwd()).resolve()
    config = {**DEFAULTS, **values}

    def absolute(value):
        if not isinstance(value, (str, PathLike)):
            raise ConfigError("Paths must be strings or pathlib paths")
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (base / path).resolve()

    for key in ("repo_root", "output_dir"):
        if config.get(key) is not None and not isinstance(config[key], (str, PathLike)):
            raise ConfigError(f"{key} must be a path string or null")
    config["repo_root"] = str(absolute(config.get("repo_root") or default_repo_root()))
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    default_output = Path(__file__).resolve().parents[1] / "results" / tag
    config["output_dir"] = str(absolute(config.get("output_dir") or default_output))
    for key in ("points", "channels", "repeats", "threads", "timeout"):
        if type(config[key]) is not int or config[key] < 1:
            raise ConfigError(f"{key} must be a positive integer")
    for key in ("warmups", "seed"):
        if type(config[key]) is not int or config[key] < 0:
            raise ConfigError(f"{key} must be a nonnegative integer")
    for key in ("synthetic", "correctness_only", "async_entropy"):
        if type(config[key]) is not bool:
            raise ConfigError(f"{key} must be a boolean")
    if not isinstance(config["device"], str) or not re.fullmatch(r"auto|cpu|cuda(?::\d+)?|cuda-sim", config["device"]):
        raise ConfigError("device must be auto, cpu, cuda, cuda:N, or cuda-sim")
    selected = config["suites"]
    if not isinstance(selected, list) or not selected or any(not isinstance(x, str) for x in selected):
        raise ConfigError("suites must be a nonempty list")
    if selected == ["all"]:
        selected = list(SUITES)
    if set(selected) - set(SUITES) or len(set(selected)) != len(selected):
        raise ConfigError("suites must contain distinct names: " + ", ".join(SUITES))
    config["suites"] = selected
    if (not isinstance(config["posq"], list) or not config["posq"] or
            any(type(q) is not int or q not in (4, 8, 16, 32, 64) for q in config["posq"]) or
            len(set(config["posq"])) != len(config["posq"])):
        raise ConfigError("posq must contain distinct values from 4,8,16,32,64")
    if config["prequantized"] not in ("auto", "yes", "no"):
        raise ConfigError("prequantized must be auto, yes, or no")
    if config["compiler"] is not None and not isinstance(config["compiler"], str):
        raise ConfigError("compiler must be a string or null")
    repo = Path(config["repo_root"])
    entries = config["files"]
    if entries is None:
        entries = [str(path) for path in sorted((repo / "data").glob("*Example/*.ply"))]
    if not isinstance(entries, list):
        raise ConfigError("files must be a list of paths or objects")
    files = []
    for entry in ([] if config["synthetic"] else entries):
        spec = dict(entry) if isinstance(entry, dict) else {"path": entry}
        if set(spec) - {"path", "prequantized", "scale", "offset"} or not isinstance(spec.get("path"), str):
            raise ConfigError("file objects accept path, prequantized, scale and offset")
        pattern = str(absolute(spec["path"]))
        paths = sorted(glob.glob(pattern)) if glob.has_magic(pattern) else [pattern]
        if not paths:
            raise ConfigError(f"No files match {pattern}")
        for name in paths:
            path = Path(name)
            if not path.is_file():
                raise ConfigError(f"Input file does not exist: {path}")
            pre = spec.get("prequantized")
            if pre is None:
                pre = config["prequantized"] == "yes" or (config["prequantized"] == "auto" and "FordExample" in path.parts)
            if type(pre) is not bool:
                raise ConfigError("file.prequantized must be boolean")
            scale, offset = spec.get("scale", .001), spec.get("offset", 131072)
            import math
            if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
                raise ConfigError("file.scale must be finite and positive")
            if type(offset) is not int:
                raise ConfigError("file.offset must be an integer")
            files.append(dict(path=str(path.resolve()), prequantized=pre, scale=scale, offset=offset))
    if len({f["path"] for f in files}) != len(files):
        raise ConfigError("Duplicate input files")
    config["files"] = files
    return config
