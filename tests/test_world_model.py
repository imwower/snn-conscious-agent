import unittest
import numpy as np

from snn_conscious.core.world_model import WorldModelConfig, WorldModelSNN


class TestWorldModelBasic(unittest.TestCase):
    def test_init_shapes(self):
        cfg = WorldModelConfig(obs_dim=8, action_dim=4, context_dim=0, hidden_dim=16, seed=42)
        wm = WorldModelSNN(cfg)

        self.assertEqual(wm.W_in.shape, (cfg.hidden_dim, cfg.obs_dim + cfg.action_dim + cfg.context_dim))
        self.assertEqual(wm.W_rec.shape, (cfg.hidden_dim, cfg.hidden_dim))
        self.assertEqual(wm.W_ro_obs.shape, (cfg.obs_dim, cfg.hidden_dim))
        self.assertEqual(wm.W_ro_rew.shape, (cfg.hidden_dim,))

    def test_step_shapes_and_reset(self):
        cfg = WorldModelConfig(obs_dim=6, action_dim=3, hidden_dim=12, seed=1)
        wm = WorldModelSNN(cfg)

        obs = np.zeros(cfg.obs_dim)
        obs[0] = 1.0
        act = np.zeros(cfg.action_dim)
        act[1] = 1.0

        pred_obs1, pred_rew1, state1 = wm.step(obs, act)
        self.assertEqual(pred_obs1.shape, (cfg.obs_dim,))
        self.assertIsInstance(pred_rew1, float)
        self.assertEqual(state1.shape, (cfg.hidden_dim,))
        self.assertEqual(wm.t, 1)

        # 重置后同一输入应得到相同的第一步分布（在相同参数下）
        wm.reset_state()
        pred_obs2, pred_rew2, state2 = wm.step(obs, act)
        np.testing.assert_allclose(pred_obs1, pred_obs2, rtol=1e-6, atol=1e-6)
        self.assertAlmostEqual(pred_rew1, pred_rew2, places=6)
        np.testing.assert_allclose(state1, state2, rtol=1e-6, atol=1e-6)


class TestWorldModelLearning(unittest.TestCase):
    def test_update_reduces_error_on_constant_target(self):
        # 构造一个简单任务：固定输入下，希望预测一个固定的目标观测与奖励
        cfg = WorldModelConfig(obs_dim=5, action_dim=2, hidden_dim=10, seed=123,
                               lr_readout=0.1, lr_hidden=0.0)  # 只训练读出，保证可学性
        wm = WorldModelSNN(cfg)

        obs = np.zeros(cfg.obs_dim)
        obs[2] = 1.0
        act = np.zeros(cfg.action_dim)
        act[0] = 1.0

        # 目标：下一步观测是一个稀疏向量，奖励为 0.5
        target_obs = np.zeros(cfg.obs_dim)
        target_obs[3] = 1.0
        target_reward = 0.5

        # 初始误差
        pred_obs, pred_rew, _ = wm.step(obs, act)
        init_mse = float(np.mean((pred_obs - target_obs) ** 2) + (pred_rew - target_reward) ** 2)

        # 多轮迭代更新（在相同输入与目标下）
        for _ in range(200):
            pred_obs, pred_rew, _ = wm.step(obs, act)
            wm.update(pred_obs, target_obs, pred_rew, target_reward)

        # 训练后误差应显著下降
        pred_obs, pred_rew, _ = wm.step(obs, act)
        final_mse = float(np.mean((pred_obs - target_obs) ** 2) + (pred_rew - target_reward) ** 2)

        self.assertLess(final_mse, init_mse * 0.5)  # 至少下降一半

    def test_invalid_shapes_raise(self):
        cfg = WorldModelConfig(obs_dim=4, action_dim=2, hidden_dim=8, seed=7)
        wm = WorldModelSNN(cfg)
        with self.assertRaises(ValueError):
            wm.step(np.zeros(3), np.zeros(cfg.action_dim))  # obs 维度不匹配
        with self.assertRaises(ValueError):
            wm.step(np.zeros(cfg.obs_dim), np.zeros(3))  # action 维度不匹配


if __name__ == "__main__":
    unittest.main()

