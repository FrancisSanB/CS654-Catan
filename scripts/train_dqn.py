#!/usr/bin/env python3
"""Train the masked DQN placement agent, save results/models/dqn.pt

python scripts/train_dqn.py --episodes 5000 --bot value
python scripts/evaluate.py --blue place:dqn:value --red value --games 1000
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catan_rl.seeding import ensure_hash_seed  # noqa: E402


def main():
    ensure_hash_seed()
    from catan_rl.dqn import DEFAULT_CKPT, train

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=5000)
    p.add_argument("--bot", default="value", choices=("value", "random"))
    p.add_argument("--placements", type=int, default=2, choices=(1, 2))
    p.add_argument("--seed", type=int, default=100_000, help="first training seed (eval uses 10000+)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--out", default=str(DEFAULT_CKPT))
    args = p.parse_args()

    agent, history = train(episodes=args.episodes, bot=args.bot, n_placements=args.placements,
                           seed=args.seed, lr=args.lr, eval_every=max(1, args.episodes // 20))
    agent.save(args.out, meta=vars(args))
    Path(args.out).with_suffix(".history.json").write_text(json.dumps(history, indent=2))
    print("saved", args.out)


if __name__ == "__main__":
    main()