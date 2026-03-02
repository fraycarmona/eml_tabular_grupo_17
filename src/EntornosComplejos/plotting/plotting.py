"""Funciones de visualización para curvas de aprendizaje y pérdida."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def _moving_average(values, window: int):
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr
    if window <= 1:
        return arr
    kernel = np.ones(window, dtype=np.float32) / float(window)
    if len(arr) < window:
        return np.array([arr.mean()], dtype=np.float32)
    return np.convolve(arr, kernel, mode="valid")


def plotlearningcurve(rewardshistory, baselinehistory=None, episode_length_history=None, window=50, title="Learning curve"):
    rewards_ma = _moving_average(rewardshistory, window)
    plt.figure(figsize=(10, 4))
    plt.plot(rewards_ma, label="Agente")

    if baselinehistory is not None:
        base_ma = _moving_average(baselinehistory, window)
        plt.plot(base_ma, label="Baseline")

    plt.title(title)
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa (media móvil)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    if episode_length_history is not None:
        len_ma = _moving_average(episode_length_history, window)
        plt.figure(figsize=(10, 3))
        plt.plot(len_ma)
        plt.title("Duración de episodios")
        plt.xlabel("Episodio")
        plt.ylabel("Pasos")
        plt.grid(alpha=0.3)
        plt.show()


def plotlosscurve(losshistory, window=50, title="Loss curve"):
    losses = np.asarray(losshistory, dtype=np.float32)
    losses = losses[np.isfinite(losses)]
    losses_ma = _moving_average(losses, window)

    plt.figure(figsize=(10, 4))
    plt.plot(losses_ma)
    plt.title(title)
    plt.xlabel("Iteración")
    plt.ylabel("Pérdida TD (media móvil)")
    plt.grid(alpha=0.3)
    plt.show()
