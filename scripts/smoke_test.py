#!/usr/bin/env python3
"""Run one seeded Catanatron game using only legal random actions.

This is a setup and integration check, not a learning experiment. It verifies
that Catanatron, Gymnasium, the environment reset/step API, legal-action access,
and result logging work together on the current machine.

Example:
    python scripts/smoke_test.py --seed 42 --max-steps 10000

The script writes a compact JSON summary to results/raw/ by default. That output
is gitignored because it is machine- and run-specific.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10_000,
        help="Safety limit on environment steps (default: 10000).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/raw"),
        help="Directory for the JSON run summary.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print the summary without writing a JSON file.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print a progress line every N steps; use 0 to disable (default: 100).",
    )
    return parser.parse_args()


def to_jsonable(value: Any) -> Any:
    """Convert common NumPy/Gymnasium values into JSON-compatible values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    return value


def get_legal_actions(env: gym.Env, info: dict[str, Any]) -> list[int]:
    """Read legal actions from the documented API, with diagnostic fallbacks."""
    valid_actions = info.get("valid_actions")
    if valid_actions is not None:
        return [int(action) for action in valid_actions]

    # Some Catanatron versions expose an action-mask helper instead.
    for target in (env, env.unwrapped):
        action_masks = getattr(target, "action_masks", None)
        if callable(action_masks):
            mask = np.asarray(action_masks(), dtype=bool)
            return np.flatnonzero(mask).astype(int).tolist()

        get_valid_actions = getattr(target, "get_valid_actions", None)
        if callable(get_valid_actions):
            return [int(action) for action in get_valid_actions()]

    raise RuntimeError(
        "Could not find legal actions. Expected info['valid_actions'], "
        "env.action_masks(), or env.unwrapped.get_valid_actions()."
    )


def main() -> int:
    args = parse_args()

    try:
        import catanatron  # noqa: F401 - importing loads package dependencies.
        import catanatron.gym  # noqa: F401 - registers Catanatron-v0 with Gymnasium.
    except ImportError as exc:
        print(
            "Catanatron is not installed. Activate your virtual environment and run:\n"
            "  python -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    random.seed(args.seed)
    np.random.seed(args.seed)
    print("Creating Catanatron environment...", flush=True)
    env = gym.make("catanatron/Catanatron-v0")

    try:
        print(f"Resetting environment with seed={args.seed}...", flush=True)
        observation, info = env.reset(seed=args.seed)
        initial_actions = get_legal_actions(env, info)
        if not initial_actions:
            raise RuntimeError("The environment returned no legal actions after reset.")
        print(
            "Environment is ready: "
            f"observation_shape={getattr(observation, 'shape', None)}, "
            f"action_space={env.action_space}, "
            f"initial_legal_actions={len(initial_actions)}",
            flush=True,
        )

        action_rng = random.Random(args.seed)
        terminated = False
        truncated = False
        reward: float | int | None = None
        step_count = 0
        legal_action_counts = [len(initial_actions)]

        while not (terminated or truncated) and step_count < args.max_steps:
            legal_actions = get_legal_actions(env, info)
            if not legal_actions:
                raise RuntimeError(f"No legal actions at step {step_count} before termination.")

            action = action_rng.choice(legal_actions)
            observation, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            if args.progress_every and step_count % args.progress_every == 0:
                print(
                    f"  step={step_count}; last_reward={reward}; "
                    f"legal_actions_before_step={len(legal_actions)}",
                    flush=True,
                )
            if not (terminated or truncated):
                legal_action_counts.append(len(get_legal_actions(env, info)))

        hit_step_limit = not (terminated or truncated)
        action_space = env.action_space
        observation_shape = getattr(observation, "shape", None)
        summary = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "environment": "catanatron/Catanatron-v0",
            "seed": args.seed,
            "action_space": str(action_space),
            "action_space_n": getattr(action_space, "n", None),
            "observation_type": type(observation).__name__,
            "observation_shape": to_jsonable(observation_shape),
            "initial_legal_action_count": len(initial_actions),
            "mean_legal_action_count": float(np.mean(legal_action_counts)),
            "min_legal_action_count": min(legal_action_counts),
            "max_legal_action_count": max(legal_action_counts),
            "steps": step_count,
            "final_reward": to_jsonable(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "hit_step_limit": hit_step_limit,
            "final_info": to_jsonable(info),
        }

        print(json.dumps(summary, indent=2, sort_keys=True, default=str))

        if not args.no_save:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            output_path = args.output_dir / f"smoke_test_seed_{args.seed}.json"
            output_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )
            print(f"\nSaved run summary to {output_path}")

        if hit_step_limit:
            print(
                f"\nWARNING: game did not end within --max-steps={args.max_steps}.",
                file=sys.stderr,
            )
            return 2
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
