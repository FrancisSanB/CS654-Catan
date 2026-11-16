"""The settlement placement enviroment."""
import random

import gymnasium as gym
import numpy as np
from catanatron.game import TURNS_LIMIT, Game
from catanatron.models.map import NUM_NODES, build_map
from catanatron.models.player import Color
from catanatron.state_functions import get_actual_victory_points
from gymnasium import spaces

from catan_rl.bots import BOTS, is_initial_settlement, owned_resources, settlements_placed

RESOURCE_TYPES = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
PORT_TYPES = RESOURCE_TYPES + [None]  # None is the 3:1 port
NODE_FEATURES = 15  # 5 pips per resource, 1 total pips, 6 port type, 3 legal/ours/opponent
GLOBAL_FEATURES = 7  # 5 resources we produce, 1 already placed one, 1 we go first


def game_stats(game, color):
    """How the game went for `color`."""
    other = next(c for c in game.state.colors if c != color)
    winner = game.winning_color()
    return {
        "win": float(winner == color),
        "loss": float(winner == other),
        "vp": get_actual_victory_points(game.state, color),
        "opp_vp": get_actual_victory_points(game.state, other),
        "turns": game.state.num_turns,
        "first": float(game.state.colors[0] == color),
    }


def node_table(catan_map):
    """Board features that never change: a (54, 12) array of pips and port type per node."""
    table = np.zeros((NUM_NODES, 12), dtype=np.float32)
    for node_id in range(NUM_NODES):
        production = catan_map.node_production[node_id]
        for r, resource in enumerate(RESOURCE_TYPES):
            table[node_id, r] = 36 * production.get(resource, 0.0) / 5  # one 6 or 8 tile = 1
        table[node_id, 5] = table[node_id, :5].sum() / 3  # a strong corner is aproximately 1
    for p, port in enumerate(PORT_TYPES):
        for node_id in catan_map.port_nodes.get(port, ()):
            table[node_id, 6 + p] = 1.0
    return table


def placement_observation(game, color, static_table):
    """The board table, plus which nodes are legal, ours or the opponent's, plus a few flags."""
    other = next(c for c in game.state.colors if c != color)
    nodes = np.zeros((NUM_NODES, 3), dtype=np.float32)
    nodes[game.state.board.buildable_node_ids(color, initial_build_phase=True), 0] = 1.0
    nodes[game.state.buildings_by_color[color]["SETTLEMENT"], 1] = 1.0
    nodes[game.state.buildings_by_color[other]["SETTLEMENT"], 2] = 1.0

    owned = owned_resources(game, color)
    flags = [float(r in owned) for r in RESOURCE_TYPES]
    flags += [float(settlements_placed(game, color) > 0), float(game.state.colors[0] == color)]
    return np.concatenate([np.hstack([static_table, nodes]).ravel(), np.array(flags, dtype=np.float32)])


class PlacementEnv(gym.Env):
    """Pick BLUE's opening settlements. A fixed bot plays everything else for both players.

    An episode has `n_placements` steps (1 or 2). The last step plays the game to the
    end: reward +1 for a win, -1 for a loss, 0 if the game hits the turn limit.
    Actions are node ids 0-53. `action_masks()` says which are legal.

    `reset(seed=s)` starts the same game as evaluation game `s`.
    """

    def __init__(self, bot="value", n_placements=2):
        self.n_placements = n_placements
        size = NUM_NODES * NODE_FEATURES + GLOBAL_FEATURES
        self.observation_space = spaces.Box(0.0, np.inf, shape=(size,), dtype=np.float32)
        self.action_space = spaces.Discrete(NUM_NODES)
        self.players = [BOTS[bot](Color.BLUE), BOTS[bot](Color.RED)]
        self.game = None
        self.static_table = None

    def action_masks(self):
        mask = np.zeros(NUM_NODES, dtype=bool)
        mask[self.game.state.board.buildable_node_ids(Color.BLUE, initial_build_phase=True)] = True
        return mask

    def _agent_to_place(self):
        game = self.game
        return (
            game.state.current_color() == Color.BLUE
            and is_initial_settlement(game, game.playable_actions)
            and settlements_placed(game, Color.BLUE) < self.n_placements
        )

    def _advance(self):
        """Let the bots play until it is the agent's turn to place. Returns True if the game ended."""
        game = self.game
        while game.winning_color() is None and game.state.num_turns < TURNS_LIMIT:
            if self._agent_to_place():
                return False
            game.play_tick()
        return True

    def _obs(self):
        return placement_observation(self.game, Color.BLUE, self.static_table)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is None:
            seed = random.randrange(2**31)
        else:
            random.seed(seed)
        catan_map = build_map("BASE")
        self.game = Game(players=self.players, seed=seed, catan_map=catan_map)
        self.static_table = node_table(catan_map)
        self._advance()
        return self._obs(), {}

    def step(self, action):
        node_id = int(action)
        settlement = next((a for a in self.game.playable_actions if a.value == node_id), None)
        if settlement is None:
            raise ValueError(f"node {node_id} is not a legal settlement")
        self.game.execute(settlement)

        if not self._advance():
            return self._obs(), 0.0, False, False, {}
        stats = game_stats(self.game, Color.BLUE)
        truncated = self.game.winning_color() is None
        return self._obs(), stats["win"] - stats["loss"], not truncated, truncated, stats
