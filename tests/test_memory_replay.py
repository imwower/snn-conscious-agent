import unittest
import numpy as np

from snn_conscious.memory.episodic_memory import EpisodicMemory, EpisodicMemoryConfig
from snn_conscious.memory.replay import ReplayLearner, ReplayConfig
from snn_conscious.core.world_model import WorldModelSNN, WorldModelConfig


class TestMemoryReplay(unittest.TestCase):
    def test_memory_add_sample(self):
        mem = EpisodicMemory(EpisodicMemoryConfig(capacity=3, seed=0))
        for i in range(5):
            obs = np.zeros(4); obs[i % 4] = 1
            act = np.zeros(2); act[i % 2] = 1
            nxt = obs.copy()
            mem.add((obs, act, nxt, 0.0, False, i))
        self.assertEqual(len(mem), 3)

    def test_replay_runs(self):
        obs_dim = 6
        act_dim = 3
        wm = WorldModelSNN(WorldModelConfig(obs_dim=obs_dim, action_dim=act_dim, hidden_dim=8, seed=0))
        mem = EpisodicMemory(EpisodicMemoryConfig(capacity=10, seed=0))
        for i in range(10):
            obs = np.zeros(obs_dim); obs[i % obs_dim] = 1
            act = np.zeros(act_dim); act[i % act_dim] = 1
            nxt = np.zeros(obs_dim); nxt[(i+1) % obs_dim] = 1
            mem.add((obs, act, nxt, 0.0, False, i))
        rl = ReplayLearner(ReplayConfig(batch_size=4, iters=2), mem, wm)
        trained = rl.run_once()
        self.assertEqual(trained, 8)


if __name__ == "__main__":
    unittest.main()

