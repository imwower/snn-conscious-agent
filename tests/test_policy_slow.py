import unittest
import numpy as np

from snn_conscious.agent.policy_slow import SlowPolicy, SlowPolicyConfig
from snn_conscious.core.world_model import WorldModelSNN, WorldModelConfig


class TestSlowPolicy(unittest.TestCase):
    def test_rollout_and_action(self):
        obs_dim = 8
        act_dim = 4
        wm = WorldModelSNN(WorldModelConfig(obs_dim=obs_dim, action_dim=act_dim, hidden_dim=16, seed=0))
        pol = SlowPolicy(SlowPolicyConfig(action_dim=act_dim, gamma=0.9, horizon=2), wm)
        obs = np.zeros(obs_dim)
        a_id, a_onehot, val = pol.act(obs)
        self.assertTrue(0 <= a_id < act_dim)
        self.assertEqual(a_onehot.shape, (act_dim,))
        self.assertIsInstance(val, float)


if __name__ == "__main__":
    unittest.main()

