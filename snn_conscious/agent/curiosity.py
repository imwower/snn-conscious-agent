"""好奇心 / 内在动机（Curiosity）

MVP：使用预测误差作为内在奖励。
- obs_intrinsic = mean(|true_obs - pred_obs|)
- rew_intrinsic = |true_rew - pred_rew|
- total = alpha * obs_intrinsic + beta * rew_intrinsic
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CuriosityConfig:
    alpha: float = 1.0
    beta: float = 0.5
    normalize: bool = False
    eps: float = 1e-8


class Curiosity:
    def __init__(self, cfg: CuriosityConfig):
        self.cfg = cfg
        self.running_mean = 0.0
        self.running_var = 1.0
        self.count = 0

    def _update_stats(self, x: float) -> None:
        self.count += 1
        delta = x - self.running_mean
        self.running_mean += delta / self.count
        delta2 = x - self.running_mean
        self.running_var += delta * delta2

    def compute(self, pred_obs: np.ndarray, true_obs: np.ndarray, pred_rew: float, true_rew: float) -> float:
        obs_err = float(np.mean(np.abs(true_obs.astype(float) - pred_obs.astype(float))))
        rew_err = abs(float(true_rew) - float(pred_rew))
        val = self.cfg.alpha * obs_err + self.cfg.beta * rew_err
        if self.cfg.normalize:
            self._update_stats(val)
            std = max((self.running_var / max(1, self.count - 1)) ** 0.5, self.cfg.eps)
            val = (val - self.running_mean) / std
        logger.debug("[Curiosity.compute] obs_err=%.4f rew_err=%.4f val=%.4f", obs_err, rew_err, val)
        return float(val)


__all__ = ["CuriosityConfig", "Curiosity"]

