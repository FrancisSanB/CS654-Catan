# Teaching AI Agents to play Catan
Aditi Naiknaware, Francis Fernandez, Kenny Melvin, Chase Cabrera

RL experiments for initial settlement placement and one simplified full-game setting.

## Experiment Infrastructure
| Item | Where |
|---|---|
| Reproducible seeds | `catan_rl/seeding.py` |
| Baseline policies | `catan_rl/bots.py` |
| Settlement placement environment | `catan_rl/envs.py` |
| Evaluation | `catan_rl/evaluate.py`, `scripts/evaluate.py` |
| Baseline experiments | `scripts/run_baselines.py` |
| Logging | `catan_rl/runlog.py`, saved to `results/` |
| Tests | `tests/test_infrastructure.py` |

## How to Run
| Task | Command |
|---|---|
| Install | `python -m pip install -r requirements.txt` |
| Install with uv (recommended) | `uv pip install -r requirements.txt` |
| Smoke test | `python scripts/smoke_test.py --seed 42` |
| Tests | `python -m pytest tests` |
| Baseline experiments | `python scripts/run_baselines.py --games 1000` |
| Evaluate one player | `python scripts/evaluate.py --blue place:rule:value --red value --games 1000` |

## Enviroment Notes
| Item | Observed |
|---|---|
| Catanatron version | 3.3.0 |
| Python version | Python 3.13.0 |
| Environment name | `catanatron/Catanatron-v0` |

| Map | MINI or BASE |
| Number of players | 1v1, 1v3, or another configuration |
| Observation type | Vector, dictionary, or other |
| Observation dimension | Measured dimension |
| Action-space type | Usually discrete |
| Total action count | Measured value |
| Valid-action representation | Exact structure |
| Reward range | Measured or documented |
| Termination condition | Exact behavior |
| Truncation condition | Exact behavior |
| Initial placement actions | How they are encoded |
| Baseline opponents | Available Catanatron players |
| Known limitations | Any issues discovered |