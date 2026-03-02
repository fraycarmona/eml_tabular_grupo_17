"""Agente SARSA semi-gradiente para Pendulum-v1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """Red MLP simple para aproximar Q(s,a)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Sequence[int] = (128, 128)) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        dims = [state_dim, *hidden_dims, action_dim]
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
            layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


@dataclass
class SarsaConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    epsilon_start: float = 0.2
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.9995
    action_bins: int = 11


class PendulumSarsaAgent:
    """SARSA con discretización de acciones para espacios continuos."""

    def __init__(self, env, seed: int = 2024, config: SarsaConfig | None = None, device: str | None = None):
        self.env = env
        self.seed = seed
        self.config = config or SarsaConfig()

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.state_dim = int(np.prod(self.env.observation_space.shape))

        low = float(self.env.action_space.low[0])
        high = float(self.env.action_space.high[0])
        self.continuous_action_values = np.linspace(low, high, self.config.action_bins, dtype=np.float32)
        self.action_dim = self.config.action_bins

        self.q_network = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.config.lr)

        self.epsilon = self.config.epsilon_start
        self.rewards_history: List[float] = []
        self.lengths_history: List[int] = []
        self.loss_history: List[float] = []

    def _state_tensor(self, state: np.ndarray) -> torch.Tensor:
        return torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _env_action(self, action_idx: int):
        return np.array([self.continuous_action_values[action_idx]], dtype=np.float32)

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        if (not greedy) and (np.random.rand() < self.epsilon):
            return int(np.random.randint(self.action_dim))

        with torch.no_grad():
            q_values = self.q_network(self._state_tensor(state))
        return int(torch.argmax(q_values, dim=1).item())

    def _sarsa_update(self, state, action_idx, reward, next_state, next_action_idx, done):
        state_t = self._state_tensor(state)
        next_state_t = self._state_tensor(next_state)

        q_sa = self.q_network(state_t)[0, action_idx]

        with torch.no_grad():
            q_next = self.q_network(next_state_t)[0, next_action_idx]
            target = torch.tensor(reward, dtype=torch.float32, device=self.device)
            if not done:
                target = target + self.config.gamma * q_next

        loss = torch.square(q_sa - target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def train(self, num_episodes: int = 5000, max_steps: int = 200):
        for episode in range(1, num_episodes + 1):
            state, _ = self.env.reset(seed=self.seed + episode)
            action_idx = self.select_action(state)
            ep_reward = 0.0
            ep_losses = []

            for step in range(1, max_steps + 1):
                next_state, reward, terminated, truncated, _ = self.env.step(self._env_action(action_idx))
                done = terminated or truncated
                next_action_idx = self.select_action(next_state) if not done else action_idx

                ep_losses.append(self._sarsa_update(state, action_idx, reward, next_state, next_action_idx, done))
                state, action_idx = next_state, next_action_idx
                ep_reward += reward
                if done:
                    break

            self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)
            self.rewards_history.append(float(ep_reward))
            self.lengths_history.append(step)
            self.loss_history.append(float(np.mean(ep_losses)))

        return {"rewards": self.rewards_history, "lengths": self.lengths_history, "losses": self.loss_history}
