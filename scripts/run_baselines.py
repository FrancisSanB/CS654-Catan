#!/usr/bin/env python3
# Runs the baseline rows of the experiment matrix.
# BLUE opens with a random node, the rule based node, or the bots own choice,
# then the same bot plays the rest of the game for both players.
#
#   python scripts/run_baselines.py --games 1000
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catan_rl.seeding import ensure_hash_seed  # noqa: E402

def main():
    ensure_hash_seed()
    from catan_rl.runlog import evaluate_and_log, format_row

    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--placements", type=int, default=2, choices=(1, 2))
    args = parser.parse_args()

    for bot in ("value", "random"):
        openings = {"random": f"place:random:{bot}", "rule": f"place:rule:{bot}", "bot": bot}
        for label, blue in openings.items():
            row = evaluate_and_log(blue, bot, games=args.games, n_placements=args.placements,
                                   name=f"baseline_{label}_{bot}_p{args.placements}")
            print(format_row(row), flush=True)


if __name__ == "__main__":
    main()
