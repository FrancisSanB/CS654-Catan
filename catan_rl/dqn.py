
"""Masked Double DQN for opening settlement placement.
Episodes are 1-2 steps long (one per settlement), so we use gamma = 1 and the
terminal +1 / -1 (or whatever reward PlacementEnv gives) is the learning signal.
"""
import os
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from catan_rl.envs import NUM_NODES, PlacementEnv, node_table, placement_observation

DEFAULT_CKPT = Path(__file__).resolve().parent.parent / "results" / "models" / "dqn.pt"


class QNet(nn.Module):
    def __init__(self, obs_dim, n_actions=NUM_NODES, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def add(self, obs, action, reward, next_obs, next_mask, done):
        self.buf.append((obs, action, reward, next_obs, next_mask, done))

    def sample(self, n):
        batch = random.sample(self.buf, n)
        obs, act, rew, nobs, nmask, done = zip(*batch)
        return (torch.as_tensor(np.stack(obs)), torch.as_tensor(act, dtype=torch.int64),
                torch.as_tensor(rew, dtype=torch.float32), torch.as_tensor(np.stack(nobs)),
                torch.as_tensor(np.stack(nmask)), torch.as_tensor(done, dtype=torch.float32))

    def __len__(self):
        return len(self.buf)


def masked_q(q, mask):
    return q.masked_fill(~mask, -torch.inf)


class DQNAgent:
    def __init__(self, obs_dim, lr=1e-4, gamma=1.0, hidden=256, device="cpu"):
        self.device = device
        self.gamma = gamma
        self.q = QNet(obs_dim, hidden=hidden).to(device)
        self.target = QNet(obs_dim, hidden=hidden).to(device)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)

    @torch.no_grad()
    def act(self, obs, mask, epsilon=0.0):
        legal = np.flatnonzero(mask)
        if random.random() < epsilon:
            return int(random.choice(legal))
        q = self.q(torch.as_tensor(obs, device=self.device).unsqueeze(0))[0]
        q = masked_q(q, torch.as_tensor(mask, device=self.device))
        return int(q.argmax())

    def update(self, batch):
        obs, act, rew, nobs, nmask, done = (t.to(self.device) for t in batch)
        q_sa = self.q(obs).gather(1, act.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Double DQN: online net picks the action, target net scores it, both masked
            best = masked_q(self.q(nobs), nmask).argmax(1, keepdim=True)
            next_q = self.target(nobs).gather(1, best).squeeze(1)
            next_q = torch.where(done.bool(), torch.zeros_like(next_q), next_q)
            target = rew + self.gamma * next_q
        loss = nn.functional.smooth_l1_loss(q_sa, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        return loss.item()

    def sync_target(self):
        self.target.load_state_dict(self.q.state_dict())

    def save(self, path, meta=None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"q": self.q.state_dict(), "obs_dim": self.q.net[0].in_features,
                    "meta": meta or {}}, path)

    @classmethod
    def load(cls, path):
        ckpt = torch.load(path, map_location="cpu")
        agent = cls(ckpt["obs_dim"])
        agent.q.load_state_dict(ckpt["q"])
        agent.target.load_state_dict(ckpt["q"])
        return agent


def train(episodes=5000, bot="value", n_placements=2, seed=100_000, batch_size=64,
          buffer_size=50_000, warmup=200, train_every=1, target_every=500,
          eps_start=1.0, eps_end=0.05, eps_frac=0.5, lr=1e-4, eval_every=500, log=print):
    """Train against `bot`. Training seeds start at 100_000 so they never overlap the
    evaluation boards (10_000 + i)."""
    env = PlacementEnv(bot=bot, n_placements=n_placements)
    obs_dim = env.observation_space.shape[0]
    agent = DQNAgent(obs_dim, lr=lr)
    buffer = ReplayBuffer(buffer_size)
    random.seed(seed)
    torch.manual_seed(seed)

    steps, returns, history = 0, deque(maxlen=200), []
    for ep in range(episodes):
        eps = max(eps_end, eps_start - (eps_start - eps_end) * ep / (eps_frac * episodes))
        obs, _ = env.reset(seed=seed + ep)
        done, ep_ret = False, 0.0
        while not done:
            mask = env.action_masks()
            action = agent.act(obs, mask, eps)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            # next mask is only meaningful mid-episode; at the end any non-empty mask works
            next_mask = mask if done else env.action_masks()
            # a turn-limit truncation (reward 0) is treated as terminal, same as a game end
            buffer.add(obs, action, reward, next_obs, next_mask, float(done))
            obs, ep_ret, steps = next_obs, ep_ret + reward, steps + 1

            if len(buffer) >= max(batch_size, warmup) and steps % train_every == 0:
                agent.update(buffer.sample(batch_size))
            if steps % target_every == 0:
                agent.sync_target()
        returns.append(ep_ret)
        if (ep + 1) % eval_every == 0:
            row = {"episode": ep + 1, "epsilon": round(eps, 3), "mean_return_200": float(np.mean(returns))}
            history.append(row)
            log(row)
    return agent, history


# ---- plugging into the existing evaluation pipeline -------------------------------
_AGENT = None


def dqn_placement(game, color, node_ids):
    """Same signature as rule_based_placement, so it works with PlacementPlayer.

    Use it from the evaluator with the spec  place:dqn:value  (checkpoint path comes
    from $DQN_CKPT, default results/models/dqn.pt)."""
    global _AGENT
    if _AGENT is None:
        torch.set_num_threads(1)
        _AGENT = DQNAgent.load(os.environ.get("DQN_CKPT", DEFAULT_CKPT))
    obs = placement_observation(game, color, node_table(game.state.board.map))
    mask = np.zeros(NUM_NODES, dtype=bool)
    mask[list(node_ids)] = True
    return _AGENT.act(obs, mask, epsilon=0.0)