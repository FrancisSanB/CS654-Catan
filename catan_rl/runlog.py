"""Saves evaluation results under results/

games/<run>.csv   - one row per game
runs/<run>.json   - settings + git commit
eval_summary.csv  - one row per run
"""
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from catan_rl.evaluate import evaluate, summarize

ROOT = Path(__file__).resolve().parent.parent
RUN_KEY = ["name", "blue", "red", "n_placements", "seed", "games"]


def _git(*args):
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None

def run_metadata():
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "uncommitted_changes": bool(_git("status", "--porcelain")),
        "catanatron": version("catanatron"),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
    }

def evaluate_and_log(blue, red, games, n_placements=2, name=None, out_dir=None):
    out_dir = Path(out_dir or ROOT / "results")
    name = name or blue
    start = time.time()
    df = evaluate(blue, red, games, n_placements=n_placements)
    row = {"name": name, "blue": blue, "red": red, "n_placements": n_placements,
           "seed": int(df["seed"].min()), **summarize(df), "seconds": round(time.time() - start, 1)}
    meta = run_metadata()

    tag = f"{name}_vs_{red}".replace("/", "_").replace(":", "_")
    (out_dir / "games").mkdir(parents=True, exist_ok=True)
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "games" / f"{tag}.csv", index=False)
    (out_dir / "runs" / f"{tag}.json").write_text(
        json.dumps({"summary": row, "meta": meta}, indent=2, default=float) + "\n")

    summary = pd.DataFrame([{**row, "git_commit": meta["git_commit"], "timestamp_utc": meta["timestamp_utc"]}])
    summary_path = out_dir / "eval_summary.csv"
    if summary_path.exists():
        # replace the old row if this setup was already ran
        old = pd.read_csv(summary_path)
        same_run = (old[RUN_KEY].astype(str) == summary[RUN_KEY].astype(str).iloc[0]).all(axis=1)
        summary = pd.concat([old[~same_run], summary], ignore_index=True)
    summary.to_csv(summary_path, index=False)
    return row

def format_row(row):
    return (f"{row['name']} vs {row['red']}: win {row['win_rate']:.3f} +- {row['win_ci95']:.3f}, "
            f"vp {row['avg_vp']:.2f}, opening pips {row['open_pips']:.1f}, "
            f"turns {row['avg_turns']:.0f} ({row['games']} games, {row['seconds']}s)")
