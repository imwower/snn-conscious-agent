import unittest
import numpy as np

from snn_conscious.envs.gridworld import GridWorld, GridConfig


class TestGridWorld(unittest.TestCase):
    def test_basic_move_and_wall(self):
        env = GridWorld(GridConfig(width=5, height=5, seed=0))
        obs = env.reset()
        x0, y0 = obs["pos"]
        # 左边是墙（外围）
        obs2, r, done, _ = env.step(3)  # left
        self.assertEqual(obs2["pos"], (x0, y0))  # 碰墙不移动
        self.assertLess(r, 0.0)

    def test_key_and_door_and_goal(self):
        env = GridWorld(GridConfig(width=5, height=5, seed=0))
        env.reset()
        # 默认布局中：S(1,1) -> 下下 到 (1,3) 拿钥匙
        env.step(2)  # down
        env.step(2)  # down
        self.assertTrue(env.has_key)
        # 回到上方，走向门 (3,1)
        env.step(1)  # up
        env.step(1)  # up -> (1,1)
        env.step(4)  # right -> (2,1)
        obs, r, done, _ = env.step(4)  # right -> 通过门到 (3,1)
        self.assertEqual(obs["pos"], (3, 1))
        # 下下到目标 (3,3)
        env.step(2)
        obs, r, done, _ = env.step(2)
        self.assertTrue(done)


if __name__ == "__main__":
    unittest.main()

