import random

import numpy as np
import pandas as pd
import pytest
from catanatron.models.player import Color

from catan_rl.bots import placement_score, random_placement, rule_based_placement
from catan_rl.envs import PlacementEnv
from catan_rl.evaluate import evaluate, make_player, play_game, summarize
from catan_rl.runlog import evaluate_and_log


def test_same_seed_gives_same_game():
    def play():
        return play_game(make_player("place:rule:value", Color.BLUE), make_player("value", Color.RED), seed=7)

    assert play() == play()

def test_rerun_replaces_summary_row(tmp_path):
    # same setup twice should only leave one row
    for games in (2, 2, 3):
        evaluate_and_log("random", "random", games=games, out_dir=tmp_path)
    summary = pd.read_csv(tmp_path / "eval_summary.csv")
    assert sorted(summary["games"]) == [2, 3]


def test_summarize():
    df = evaluate("place:rule:random", "random", games=4, workers=1)
    s = summarize(df)
    assert s["games"] == 4
    assert 0 <= s["win_rate"] <= 1
    assert s["open_pips"] >= s["first_pips"] > 0


def test_bad_spec():
    with pytest.raises(ValueError):
        evaluate("place:nope:value", "value", games=1, workers=1)


def test_placement_rules_pick_legal_nodes():
    env = PlacementEnv(bot="random")
    env.reset(seed=3)
    game = env.game
    legal = np.flatnonzero(env.action_masks()).tolist()

    assert random_placement(game, Color.BLUE, legal) in legal

    # rule should pick the best scoring node
    rule = rule_based_placement(game, Color.BLUE, legal)
    best = max(placement_score(game, Color.BLUE, n) for n in legal)
    assert rule in legal
    assert placement_score(game, Color.BLUE, rule) == best


@pytest.mark.parametrize("n_placements", [1, 2])
def test_env_episode(n_placements):
    env = PlacementEnv(bot="random", n_placements=n_placements)
    obs, _ = env.reset(seed=11)
    assert obs.shape == (54 * 15 + 7,)
    assert obs.dtype == np.float32

    rng = random.Random(0)
    for step in range(n_placements):
        legal = np.flatnonzero(env.action_masks()).tolist()
        obs, reward, terminated, truncated, info = env.step(rng.choice(legal))
        if step < n_placements - 1:
            # game shouldnt be over untill the last placement
            assert reward == 0.0
            assert not terminated and not truncated

    assert terminated or truncated
    assert reward == info["win"] - info["loss"]


def test_env_rejects_illegal_node():
    env = PlacementEnv(bot="random")
    env.reset(seed=5)
    node = int(np.flatnonzero(env.action_masks())[0])
    env.step(node)
    with pytest.raises(ValueError):
        env.step(node)  # node is already occupied

def test_reset_seed_repeats_board():
    env = PlacementEnv(bot="random")
    a, _ = env.reset(seed=21)
    b, _ = env.reset(seed=21)
    c, _ = env.reset(seed=22)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


# the env and the evaluation code should play the exact same game for a seed
@pytest.mark.parametrize("n_placements", [1, 2])
def test_env_matches_evaluation_game(n_placements):
    keys = ("win", "loss", "vp", "opp_vp", "turns")
    for seed in (10_000, 10_001, 10_002):
        env = PlacementEnv(bot="value", n_placements=n_placements)
        env.reset(seed=seed)
        done = False
        while not done:
            legal = np.flatnonzero(env.action_masks()).tolist()
            node = rule_based_placement(env.game, Color.BLUE, legal)
            _, _, terminated, truncated, info = env.step(node)
            done = terminated or truncated

        blue = make_player("place:rule:value", Color.BLUE, n_placements)
        expected = play_game(blue, make_player("value", Color.RED), seed)
        for k in keys:
            assert info[k] == expected[k], f"{k} differs for seed {seed}"
