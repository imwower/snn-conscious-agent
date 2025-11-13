import unittest
import numpy as np

from snn_conscious.agent.policy_fast import FastPolicy, FastPolicyConfig
from snn_conscious.core.world_model import WorldModelSNN, WorldModelConfig


class TestFastPolicy(unittest.TestCase):
    def test_act_returns_valid_action(self):
        obs_dim = 10
        act_dim = 5
        wm = WorldModelSNN(WorldModelConfig(obs_dim=obs_dim, action_dim=act_dim, hidden_dim=8, seed=0))
        pol = FastPolicy(FastPolicyConfig(action_dim=act_dim, epsilon=0.0), wm)
        obs = np.zeros(obs_dim)
        a_id, a_onehot, pred_rew = pol.act(obs)
        self.assertTrue(0 <= a_id < act_dim)
        self.assertEqual(a_onehot.shape, (act_dim,))
        self.assertEqual(float(np.sum(a_onehot)), 1.0)


if __name__ == "__main__":
    unittest.main()

