# Baseline policies - catanatron bots and two ways of picking an opening settlment

import random

from catanatron.models.actions import ActionType
from catanatron.models.enums import SETTLEMENT
from catanatron.models.player import Player, RandomPlayer
from catanatron.players.value import ValueFunctionPlayer

BOTS = {"random": RandomPlayer, "value": ValueFunctionPlayer}


def node_pips(catan_map, node_id):
    # dots on the number tokens around a node (a 6 or 8 has 5 dots)
    return 36 * sum(catan_map.node_production[node_id].values())


def owned_resources(game, color):
    resources = set()
    for node_id in game.state.buildings_by_color[color][SETTLEMENT]:
        resources.update(game.state.board.map.node_production[node_id])
    return resources


def placement_score(game, color, node_id):
    """Pips, plus 2 for each resource type the player does not have yet."""
    catan_map = game.state.board.map
    new_types = set(catan_map.node_production[node_id]) - owned_resources(game, color)
    return node_pips(catan_map, node_id) + 2 * len(new_types)


def rule_based_placement(game, color, node_ids):
    scores = {n: placement_score(game, color, n) for n in node_ids}
    best = max(scores.values())
    return random.choice(sorted(n for n in node_ids if scores[n] == best))


def random_placement(game, color, node_ids):
    return random.choice(sorted(node_ids))


def dqn_placement(game, color, node_ids):
    # imported lazily: dqn.py imports envs.py, which imports this file
    from catan_rl.dqn import dqn_placement as pick
    return pick(game, color, node_ids)


PLACEMENTS = {"rule": rule_based_placement, "random": random_placement, "dqn": dqn_placement}


def settlements_placed(game, color):
    return len(game.state.buildings_by_color[color][SETTLEMENT])


def is_initial_settlement(game, playable_actions):
    return game.state.is_initial_build_phase and playable_actions[0].action_type == ActionType.BUILD_SETTLEMENT


class PlacementPlayer(Player):
    """Bot whose opening settlements are picked by a placement rule.
    The rule picks the first n_placements settlements, the bot does everything else."""

    def __init__(self, color, base, placement, n_placements=2):
        super().__init__(color)
        self.base = base
        self.placement = placement
        self.n_placements = n_placements

    def decide(self, game, playable_actions):
        if is_initial_settlement(game, playable_actions) and settlements_placed(game, self.color) < self.n_placements:
            node_id = self.placement(game, self.color, [a.value for a in playable_actions])
            return next(a for a in playable_actions if a.value == node_id)
        return self.base.decide(game, playable_actions)