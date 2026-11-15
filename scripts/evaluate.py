#!/usr/bin/env python3
"""Evaluate one player against an opponent, results go to results/

python scripts/evaluate.py --blue place:rule:value --red value --games 1000
"""
import argparse
import sys
from pathlib import Path

# so we can import catan_rl
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catan_rl.seeding import ensure_hash_seed  # noqa: E402

def main():
    ensure_hash_seed()
    from catan_rl.runlog import evaluate_and_log, format_row

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--blue", required=True, help="player being tested")
    p.add_argument("--red", required=True, help="oponent")
    p.add_argument("--games", type=int, default=1000)
    p.add_argument("--placements", type=int, default=2, choices=(1, 2), help="settlements the rule picks")
    p.add_argument("--out-dir")
    args = p.parse_args()

    row = evaluate_and_log(args.blue, args.red, games=args.games, n_placements=args.placements,
                           out_dir=args.out_dir)
    print(format_row(row))


if __name__ == "__main__":
    main()
