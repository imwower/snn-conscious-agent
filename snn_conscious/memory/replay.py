"""记忆重放（Replay）MVP 实现。

职责：
- 从情景记忆中抽样，并调用世界模型进行离线学习；
- 支持配置每次重放的样本数与步数；
- 记录基本统计（学习次数等）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import logging
import numpy as np

from .episodic_memory import EpisodicMemory
from snn_conscious.core.world_model import WorldModelSNN

logger = logging.getLogger(__name__)


@dataclass
class ReplayConfig:
    batch_size: int = 16
    iters: int = 1


class ReplayLearner:
    def __init__(self, cfg: ReplayConfig, memory: EpisodicMemory, wm: WorldModelSNN):
        self.cfg = cfg
        self.memory = memory
        self.wm = wm
        self.total_updates = 0

    def run_once(self) -> int:
        """执行一次离线重放（共 iters 批次）。返回实际训练样本数。"""
        trained = 0
        for _ in range(self.cfg.iters):
            batch = self.memory.sample(self.cfg.batch_size)
            for obs, a_onehot, next_obs, reward, done, t in batch:
                self.wm.learn_from_sample(obs, a_onehot, next_obs, reward)
                trained += 1
                self.total_updates += 1
        logger.info("[Replay.run_once] samples=%d total_updates=%d", trained, self.total_updates)
        return trained


__all__ = ["ReplayConfig", "ReplayLearner"]

