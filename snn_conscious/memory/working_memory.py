"""工作记忆（Working Memory）MVP 实现。

职责：
- 保存最近若干步“意识内容”的临时存储；
- 支持指数衰减的聚合读出（作为当前工作记忆表示）；
- 可用于与 GWS 联动：点火时将被选内容写入工作记忆。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WorkingMemoryConfig:
    content_dim: int
    capacity: int = 32
    decay: float = 0.8  # 聚合读出时的指数衰减因子（越接近1越长记忆）


@dataclass
class WMItem:
    vector: np.ndarray
    source: str
    t: int
    strength: float = 1.0


class WorkingMemory:
    def __init__(self, cfg: WorkingMemoryConfig):
        if cfg.content_dim <= 0 or cfg.capacity <= 0:
            raise ValueError("content_dim/capacity 必须为正")
        if not (0.0 < cfg.decay <= 1.0):
            raise ValueError("decay 需在 (0,1] 区间")
        self.cfg = cfg
        self.buffer: List[WMItem] = []

    def reset(self) -> None:
        self.buffer.clear()
        logger.info("[WorkingMemory.reset] 已清空")

    def add(self, item: WMItem) -> None:
        if item.vector.shape != (self.cfg.content_dim,):
            raise ValueError("向量维度不匹配")
        if len(self.buffer) >= self.cfg.capacity:
            self.buffer.pop(0)
        self.buffer.append(item)

    def read_vector(self) -> np.ndarray:
        """返回指数衰减聚合的向量表示（长度 content_dim）。"""
        if not self.buffer:
            return np.zeros(self.cfg.content_dim, dtype=float)
        v = np.zeros(self.cfg.content_dim, dtype=float)
        w = 1.0
        for item in reversed(self.buffer):
            v += w * item.strength * item.vector
            w *= self.cfg.decay
        return v

    def last(self) -> Optional[WMItem]:
        return self.buffer[-1] if self.buffer else None


__all__ = ["WorkingMemoryConfig", "WorkingMemory", "WMItem"]

