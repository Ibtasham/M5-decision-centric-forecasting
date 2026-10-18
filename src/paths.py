"""Central path resolution. Never hard-code absolute workspace paths in this
project: they are rewritten by the execution harness. Derive everything from
the ARENA_WORKSPACE environment variable (falls back to the repo location)."""
import os
from pathlib import Path

# Project root = the directory containing this file's parent ("src").
# Prefer the ARENA_WORKSPACE environment variable only when it actually points
# at this project, so an unpacked release copy resolves its own paths even if
# the variable is still set to a different workspace.
_ROOT = Path(__file__).resolve().parents[1]
_env = os.environ.get("ARENA_WORKSPACE")

ROOT = _ROOT
RAW = Path(os.environ.get("M5INV_DATA", _ROOT / "data"))
PROC = RAW / "proc"
RESULTS = ROOT / "results"
FIGS = ROOT / "figs"
LOGS = ROOT / "logs"

for _d in (PROC, RESULTS, FIGS, LOGS):
    _d.mkdir(parents=True, exist_ok=True)

N_DAYS = 1969
DAY_COLS = [f"d_{i}" for i in range(1, N_DAYS + 1)]
ID_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
