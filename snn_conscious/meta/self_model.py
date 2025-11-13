"""自我模型 / 元认知（Self-Model & Metacognition）MVP 实现。

职责（MVP）：
- 根据世界模型的高层状态表示 h 与近期误差，估计“当前置信度/风险”；
- 输出：confidence ∈ (0,1)、risk = 1-confidence、should_think（当 risk 超阈值时 True）；
- 学习：对给定的二分类目标（成功=1/失败=0）执行逻辑回归的增量更新（交叉熵梯度）。

说明：
- 本实现使用线性+sigmoid 的形式，便于理解与可控；
- 未来可替换为 SNN/LIF 神经元并采用三因子规则；
- 输入特征可扩展（例如加入任务上下文、误差的时间窗口统计等）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SelfModelConfig:
    input_dim: int
    lr: float = 0.05
    think_threshold: float = 0.5  # 风险阈值（risk>阈值则 should_think=True）
    l2: float = 0.0
    clip: float = 5.0
    seed: Optional[int] = None


class SelfModel:
    """简单的逻辑回归自我模型。"""

    def __init__(self, cfg: SelfModelConfig):
        if cfg.input_dim <= 0:
            raise ValueError("input_dim 必须为正")
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.w = self.rng.normal(0.0, 0.1, size=(cfg.input_dim,))
        self.b = 0.0

    @staticmethod
    def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
        return 1.0 / (1.0 + np.exp(-x))

    def estimate(self, x: np.ndarray) -> Tuple[float, float, bool]:
        """估计 (confidence, risk, should_think)。

        x: 输入特征（长度 input_dim）。
        """
        if x.shape != (self.cfg.input_dim,):
            raise ValueError(f"特征维度应为 ({self.cfg.input_dim},), 实际为 {x.shape}")
        z = float(np.dot(self.w, x) + self.b)
        conf = float(self._sigmoid(z))  # 置信度
        risk = 1.0 - conf
        should = risk > self.cfg.think_threshold
        logger.debug("[SelfModel.estimate] conf=%.3f risk=%.3f should=%s", conf, risk, should)
        return conf, risk, bool(should)

    def update(self, x: np.ndarray, target_success: int) -> None:
        """交叉熵增量更新。

        target_success: 1 表示该情境最终成功（或高把握），0 表示失败（或低把握）。
        """
        if x.shape != (self.cfg.input_dim,):
            raise ValueError("输入维度不匹配")
        y = 1.0 if target_success else 0.0
        z = float(np.dot(self.w, x) + self.b)
        p = float(self._sigmoid(z))
        # 交叉熵对线性项梯度： (p - y)
        grad = (p - y)
        self.w -= self.cfg.lr * (grad * x + self.cfg.l2 * self.w)
        self.b -= self.cfg.lr * (grad + self.cfg.l2 * self.b)
        # 裁剪
        if self.cfg.clip and self.cfg.clip > 0:
            np.clip(self.w, -self.cfg.clip, self.cfg.clip, out=self.w)
            self.b = float(np.clip(self.b, -self.cfg.clip, self.cfg.clip))
        logger.debug("[SelfModel.update] y=%.1f p=%.3f grad=%.3f", y, p, grad)


__all__ = ["SelfModelConfig", "SelfModel"]

