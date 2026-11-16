"""Plays games between two players and records the results.

Player specs:
    random, value                       catanatron bots
    place:rule:BOT, place:random:BOT    BOT plays but the rule picks the opening

BLUE is the player we are testing, RED is the oponent. Game i uses seed + i so
every BLUE player gets the same boards
"""
import random
from multiprocessing import Pool

import numpy as np
import pandas as pd
from catanatron.game import Game
from catanatron.models.actions import ActionType
from catanatron.models.map import build_map
from catanatron.models.player import Color

from catan_rl.bots import BOTS, PLACEMENTS, PlacementPlayer, node_pips
from catan_rl.envs import game_stats


def make_player(spec, color, n_placements=2):
    if spec in BOTS:
        return BOTS[spec](color)
    parts = spec.split(":")
    if len(parts) == 3 and parts[0] == "place" and parts[1] in PLACEMENTS and parts[2] in BOTS:
        return PlacementPlayer(color, BOTS[parts[2]](color), PLACEMENTS[parts[1]], n_placements)
    raise ValueError(f"unknown player spec {spec!r}")


def opening_stats(game, color):
    nodes = [r.action.value for r in game.state.action_records
             if r.action.color == color and r.action.action_type == ActionType.BUILD_SETTLEMENT][:2]
    catan_map = game.state.board.map
    return {
        "first_node": nodes[0],
        "second_node": nodes[1],
        "first_pips": node_pips(catan_map, nodes[0]),
        "open_pips": sum(node_pips(catan_map, n) for n in nodes),
        "open_resources": len(set().union(*(catan_map.node_production[n] for n in nodes))),
    }


def play_game(blue, red, seed):
    random.seed(seed)
    game = Game([blue, red], seed=seed, catan_map=build_map("BASE"))
    game.play()
    return {"seed": seed, **game_stats(game, Color.BLUE), **opening_stats(game, Color.BLUE)}


def _play_spec_game(job):
    blue, red, seed, n_placements = job
    return play_game(make_player(blue, Color.BLUE, n_placements),
                     make_player(red, Color.RED, n_placements), seed)


def evaluate(blue, red, games=100, seed=10_000, workers=8, n_placements=2):
    # fail early on a bad spec
    make_player(blue, Color.BLUE)
    make_player(red, Color.RED)

    jobs = [(blue, red, seed + i, n_placements) for i in range(games)]
    if workers <= 1:
        rows = [_play_spec_game(job) for job in jobs]
    else:
        with Pool(workers) as pool:
            rows = pool.map(_play_spec_game, jobs)
    return pd.DataFrame(rows).assign(blue=blue, red=red)


def summarize(df):
    n = len(df)
    p = df["win"].mean()
    # 95% interval on the win rate
    return {
        "games": n,
        "win_rate": p,
        "win_ci95": 1.96 * np.sqrt(p * (1 - p) / n),
        "avg_vp": df["vp"].mean(),
        "avg_opp_vp": df["opp_vp"].mean(),
        "avg_turns": df["turns"].mean(),
        "truncated": ((df["win"] == 0) & (df["loss"] == 0)).mean(),
        "first_pips": df["first_pips"].mean(),
        "open_pips": df["open_pips"].mean(),
        "open_resources": df["open_resources"].mean(),
    }
