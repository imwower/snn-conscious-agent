"""快速策略（fast policy）

MVP 设计：
- 使用世界模型的 predict 接口，对所有动作 one-hot 进行评估，选择预测奖励最大的动作；
- 支持 epsilon-greedy 探索；
- 返回离散动作与动作 one-hot 编码。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import logging
import numpy as np

from snn_conscious.core.world_model import WorldModelSNN

logger = logging.getLogger(__name__)


@dataclass
class FastPolicyConfig:
    action_dim: int
    epsilon: float = 0.1


class FastPolicy:
    def __init__(self, cfg: FastPolicyConfig, wm: WorldModelSNN):
        if cfg.action_dim <= 0:
            raise ValueError("action_dim 必须为正整数")
        self.cfg = cfg
        self.wm = wm
        self.rng = np.random.default_rng()

    def _one_hot(self, a: int) -> np.ndarray:
        v = np.zeros(self.cfg.action_dim, dtype=float)
        if 0 <= a < self.cfg.action_dim:
            v[a] = 1.0
        return v

    def act(self, obs_vec: np.ndarray) -> Tuple[int, np.ndarray, float]:
        """根据当前观测向量选择动作。

        返回：(action_id, action_one_hot, predicted_reward_of_choice)
        """
        # epsilon 探索
        if self.rng.random() < self.cfg.epsilon:
            a = int(self.rng.integers(self.cfg.action_dim))
            a_onehot = self._one_hot(a)
            _, pred_rew, _ = self.wm.predict(obs_vec, a_onehot)
            logger.debug("[FastPolicy.act] epsilon action=%d pred_rew=%.3f", a, pred_rew)
            return a, a_onehot, float(pred_rew)

        # 评估所有动作
        preds = []
        for a in range(self.cfg.action_dim):
            a_onehot = self._one_hot(a)
            _, pred_rew, _ = self.wm.predict(obs_vec, a_onehot)
            preds.append((a, float(pred_rew)))

        # 按预测奖励选择
        a_best, rew_best = max(preds, key=lambda t: t[1])
        logger.debug("[FastPolicy.act] greedy action=%d pred_rew=%.3f", a_best, rew_best)
        return a_best, self._one_hot(a_best), rew_best


__all__ = ["FastPolicyConfig", "FastPolicy"]

