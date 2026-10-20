"""Run from any working directory, including an extracted distribution."""
import sys

sys.dont_write_bytecode = True
from elic_bench.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
