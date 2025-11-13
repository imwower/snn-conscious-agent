"""慢速策略（slow policy）

MVP 设计：
- 使用世界模型的 predict 进行有限视野的滚动评估（horizon>=1），
  以折扣因子 gamma 聚合未来奖励；
- 返回累计预测奖励最大的动作；
- 可用于在自我模型提示“应当多思考”时调用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import logging
import numpy as np

from snn_conscious.core.world_model import WorldModelSNN

logger = logging.getLogger(__name__)


@dataclass
class SlowPolicyConfig:
    action_dim: int
    gamma: float = 0.9
    horizon: int = 2


class SlowPolicy:
    def __init__(self, cfg: SlowPolicyConfig, wm: WorldModelSNN):
        if cfg.action_dim <= 0:
            raise ValueError("action_dim 必须为正整数")
        if cfg.horizon <= 0:
            raise ValueError("horizon 必须 >= 1")
        self.cfg = cfg
        self.wm = wm

    def _one_hot(self, a: int) -> np.ndarray:
        v = np.zeros(self.cfg.action_dim, dtype=float)
        if 0 <= a < self.cfg.action_dim:
            v[a] = 1.0
        return v

    def _rollout_reward(self, obs_vec: np.ndarray, a0: int) -> float:
        # step 1
        a = a0
        a_onehot = self._one_hot(a)
        pred_obs, pred_rew, h = self.wm.predict(obs_vec, a_onehot)
        ret = float(pred_rew)
        obs = pred_obs
        # 后续 horizon-1 步：贪心展开
        for k in range(1, self.cfg.horizon):
            # 在预测的 obs 上再评估所有动作，选 best
            best_val = -1e18
            for a2 in range(self.cfg.action_dim):
                _, pred_rew2, _ = self.wm.predict(obs, self._one_hot(a2))
                if float(pred_rew2) > best_val:
                    best_val = float(pred_rew2)
                    best_a2 = a2
            # 应用 best 动作得到下一状态
            obs, rew2, _ = self.wm.predict(obs, self._one_hot(best_a2))
            ret += (self.cfg.gamma ** k) * float(rew2)
        return ret

    def act(self, obs_vec: np.ndarray) -> Tuple[int, np.ndarray, float]:
        scores = []
        for a in range(self.cfg.action_dim):
            score = self._rollout_reward(obs_vec, a)
            scores.append((a, score))
        a_best, v_best = max(scores, key=lambda t: t[1])
        logger.debug("[SlowPolicy.act] action=%d rollout=%.3f", a_best, v_best)
        return a_best, self._one_hot(a_best), float(v_best)


__all__ = ["SlowPolicyConfig", "SlowPolicy"]

