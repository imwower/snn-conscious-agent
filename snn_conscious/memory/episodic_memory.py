"""情景记忆（Episodic Memory）MVP 实现。

职责：
- 存储 (obs, action_onehot, next_obs, reward, done, t) 序列；
- 支持随机采样 batch；
- 提供清理与容量限制（FIFO）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)


Transition = Tuple[np.ndarray, np.ndarray, np.ndarray, float, bool, int]


@dataclass
class EpisodicMemoryConfig:
    capacity: int = 10000
    seed: Optional[int] = None


class EpisodicMemory:
    def __init__(self, cfg: EpisodicMemoryConfig):
        if cfg.capacity <= 0:
            raise ValueError("capacity 必须为正整数")
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.data: List[Transition] = []
        self.ptr: int = 0

    def __len__(self) -> int:
        return len(self.data)

    def add(self, tr: Transition) -> None:
        if len(self.data) < self.cfg.capacity:
            self.data.append(tr)
        else:
            self.data[self.ptr] = tr
            self.ptr = (self.ptr + 1) % self.cfg.capacity

    def sample(self, batch_size: int) -> List[Transition]:
        if batch_size <= 0:
            return []
        n = min(batch_size, len(self.data))
        idx = self.rng.choice(len(self.data), size=n, replace=False)
        return [self.data[i] for i in idx]

    def clear(self) -> None:
        self.data.clear()
        self.ptr = 0
        logger.info("[EpisodicMemory.clear] 记忆已清空")


__all__ = ["EpisodicMemoryConfig", "EpisodicMemory", "Transition"]

