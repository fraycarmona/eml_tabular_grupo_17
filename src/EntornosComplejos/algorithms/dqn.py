"""Rutina de actualización DQN."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def dqnupdate(q_network, target_network, optimizer, batch, gamma: float, device: torch.device):
    """Ejecuta una actualización de DQN con pérdida MSE."""
    states, actions, rewards, next_states, dones = batch

    states_t = torch.tensor(states, dtype=torch.float32, device=device)
    actions_t = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    next_states_t = torch.tensor(next_states, dtype=torch.float32, device=device)
    dones_t = torch.tensor(dones, dtype=torch.float32, device=device)

    q_values = q_network(states_t).gather(1, actions_t).squeeze(1)

    with torch.no_grad():
        next_q_values = target_network(next_states_t).max(dim=1).values
        targets = rewards_t + gamma * next_q_values * (1.0 - dones_t)

    loss = F.mse_loss(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
