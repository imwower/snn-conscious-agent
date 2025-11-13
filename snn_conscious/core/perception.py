"""感知编码（Perception Encoder）

职责：
- 将环境的原始状态（例如 GridWorld 中的位置、邻近格子、是否持有钥匙等）
  编码为固定维度的向量形式，供下游 SNN 使用。

MVP 设计：
- 针对简化 GridWorld：
  - 位置 one-hot（W*H）；
  - 邻近细胞类型 one-hot（上/下/左/右，各 6 类：free/wall/key/door/goal/out）；
  - 持有钥匙标志（1 维）。

注意：
- 该模块仅做纯编码，不持久化内部状态；
- 若后续引入更复杂传感器，可在此扩展。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List
import logging
import numpy as np

logger = logging.getLogger(__name__)


CELL_TYPES = ["free", "wall", "key", "door", "goal", "out"]


@dataclass
class PerceptionConfig:
    width: int
    height: int

    def obs_dim(self) -> int:
        # 位置 one-hot + 四方向邻近(每个6类) + has_key 1bit
        return self.width * self.height + 4 * len(CELL_TYPES) + 1


class PerceptionEncoder:
    """将 GridWorld 状态编码为固定维度向量。"""

    def __init__(self, cfg: PerceptionConfig):
        if cfg.width <= 0 or cfg.height <= 0:
            raise ValueError("width/height 必须为正整数")
        self.cfg = cfg

    def _pos_one_hot(self, pos: Tuple[int, int]) -> np.ndarray:
        x, y = pos
        idx = y * self.cfg.width + x
        vec = np.zeros(self.cfg.width * self.cfg.height, dtype=float)
        if 0 <= idx < vec.size:
            vec[idx] = 1.0
        return vec

    @staticmethod
    def _cell_one_hot(cell: str) -> np.ndarray:
        vec = np.zeros(len(CELL_TYPES), dtype=float)
        try:
            vec[CELL_TYPES.index(cell)] = 1.0
        except ValueError:
            # 未知类型视作 out
            vec[CELL_TYPES.index("out")] = 1.0
        return vec

    def encode(self, state: Dict) -> np.ndarray:
        """编码状态为向量。

        期望的 state 字段：
        - pos: (x, y)
        - neighbors: {"up": type, "down": type, "left": type, "right": type}
        - has_key: bool
        """
        pos_vec = self._pos_one_hot(tuple(state["pos"]))
        up = self._cell_one_hot(state["neighbors"].get("up", "out"))
        down = self._cell_one_hot(state["neighbors"].get("down", "out"))
        left = self._cell_one_hot(state["neighbors"].get("left", "out"))
        right = self._cell_one_hot(state["neighbors"].get("right", "out"))
        has_key = np.array([1.0 if state.get("has_key", False) else 0.0], dtype=float)

        obs = np.concatenate([pos_vec, up, down, left, right, has_key])
        assert obs.shape[0] == self.cfg.obs_dim(), "编码维度与配置不一致"
        return obs


__all__ = ["PerceptionConfig", "PerceptionEncoder", "CELL_TYPES"]

