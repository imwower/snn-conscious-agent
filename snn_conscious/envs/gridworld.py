"""简化 GridWorld 环境（MVP）。

功能：
- 小型网格；包含墙(#)、钥匙(K)、门(D)、目标(G)、空地(.)；
- agent 起点 S；持有钥匙后可通过门；到达目标获得奖励并结束。
- 动作：0 原地、1 上、2 下、3 左、4 右。

观测：
- 提供原始状态字典：pos、neighbors、has_key；
- 由 PerceptionEncoder 负责转为向量。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)


Action = int


@dataclass
class GridConfig:
    width: int = 5
    height: int = 5
    step_cost: float = -0.01
    wall_bump_cost: float = -0.02
    goal_reward: float = 1.0
    key_reward: float = 0.1
    door_open_cost: float = -0.0
    seed: Optional[int] = None


class GridWorld:
    """简化 GridWorld。

    地图字符：
      '.' 空地, '#' 墙, 'K' 钥匙, 'D' 门, 'G' 目标, 'S' 起点
    """

    def __init__(self, cfg: GridConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.reset()

    # ------------------------- 地图构造与重置 -------------------------
    def _default_map(self) -> np.ndarray:
        w, h = self.cfg.width, self.cfg.height
        grid = np.full((h, w), '.', dtype='<U1')
        # 周围围墙
        grid[0, :] = '#'
        grid[-1, :] = '#'
        grid[:, 0] = '#'
        grid[:, -1] = '#'
        # 放置 K, D, G, S（固定布局，便于测试）
        # S 在 (1,1), K 在 (1,3), D 在 (3,1), G 在 (3,3)
        grid[1, 1] = 'S'
        grid[3, 1] = 'K' if h > 4 else 'K'
        grid[1, 3] = 'D'
        grid[3, 3] = 'G'
        return grid

    def reset(self) -> Dict:
        self.grid = self._default_map()
        self.has_key = False
        self.done = False
        self.t = 0
        self.agent_pos = self._find_char('S')
        logger.info("[GridWorld.reset] pos=%s has_key=%s", self.agent_pos, self.has_key)
        return self._obs()

    def _find_char(self, ch: str) -> Tuple[int, int]:
        ys, xs = np.where(self.grid == ch)
        if len(xs) == 0:
            raise ValueError(f"地图中缺少字符 {ch}")
        return int(xs[0]), int(ys[0])

    # ------------------------------ 观测 ------------------------------
    def _cell_type(self, x: int, y: int) -> str:
        if x < 0 or x >= self.cfg.width or y < 0 or y >= self.cfg.height:
            return "out"
        ch = self.grid[y, x]
        if ch == '#':
            return "wall"
        if ch == '.':
            return "free"
        if ch == 'S':
            return "free"  # 起点视为空地
        if ch == 'K':
            return "key"
        if ch == 'D':
            return "door"
        if ch == 'G':
            return "goal"
        return "free"

    def _neighbors(self, x: int, y: int) -> Dict[str, str]:
        return {
            "up": self._cell_type(x, y - 1),
            "down": self._cell_type(x, y + 1),
            "left": self._cell_type(x - 1, y),
            "right": self._cell_type(x + 1, y),
        }

    def _obs(self) -> Dict:
        x, y = self.agent_pos
        return {
            "pos": (x, y),
            "neighbors": self._neighbors(x, y),
            "has_key": self.has_key,
            "t": self.t,
        }

    # ------------------------------ 动作 ------------------------------
    @staticmethod
    def action_dim() -> int:
        return 5  # stay, up, down, left, right

    def step(self, action: Action) -> Tuple[Dict, float, bool, Dict]:
        if self.done:
            return self._obs(), 0.0, True, {"msg": "already done"}

        self.t += 1
        x, y = self.agent_pos
        dx, dy = 0, 0
        if action == 1:  # up
            dy = -1
        elif action == 2:  # down
            dy = 1
        elif action == 3:  # left
            dx = -1
        elif action == 4:  # right
            dx = 1
        # else 0: stay

        nx, ny = x + dx, y + dy
        cell = self._cell_type(nx, ny)
        reward = self.cfg.step_cost

        # 碰墙
        if cell == "wall" or cell == "out":
            reward += self.cfg.wall_bump_cost
            nx, ny = x, y
        # 门
        elif cell == "door":
            if self.has_key:
                reward += self.cfg.door_open_cost
                # 通过门：把门变为空地
                self.grid[ny, nx] = '.'
            else:
                # 无钥匙视作墙
                reward += self.cfg.wall_bump_cost
                nx, ny = x, y
        # 钥匙
        elif cell == "key":
            self.has_key = True
            reward += self.cfg.key_reward
            self.grid[ny, nx] = '.'
        # 目标
        elif cell == "goal":
            reward += self.cfg.goal_reward
            self.done = True

        # 更新位置（清理旧 S 标记）
        self.grid[y, x] = '.'
        self.agent_pos = (nx, ny)
        self.grid[ny, nx] = 'S'

        obs = self._obs()
        info = {}
        if self.done:
            info["terminal"] = True
        return obs, reward, self.done, info


__all__ = ["GridWorld", "GridConfig"]

