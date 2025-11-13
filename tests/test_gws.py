import unittest
import numpy as np

from snn_conscious.core.gws import (
    GlobalWorkspaceConfig,
    GlobalWorkspace,
    GWSContent,
)


class TestGWSBasic(unittest.TestCase):
    def test_argmax_selection_and_ignition(self):
        cfg = GlobalWorkspaceConfig(content_dim=4, ignition_threshold=0.5, temperature=1.0,
                                    stochastic=False, topk_log_k=2)
        gws = GlobalWorkspace(cfg)

        cands = [
            GWSContent(vector=np.ones(4), source="A", priority=0.2),
            GWSContent(vector=np.arange(4), source="B", priority=0.8),
            GWSContent(vector=-np.ones(4), source="C", priority=0.3),
        ]

        sel, ign, strength, info = gws.step(t=0, candidates=cands)
        self.assertIsNotNone(sel)
        self.assertEqual(sel.source, "B")
        self.assertTrue(ign)
        self.assertAlmostEqual(strength, 0.8 - 0.5, places=6)
        self.assertEqual(info.selected_idx, 1)
        self.assertEqual(len(info.priorities), 3)
        self.assertEqual(gws.stats["ignitions"], 1)

    def test_empty_candidates(self):
        cfg = GlobalWorkspaceConfig(content_dim=2)
        gws = GlobalWorkspace(cfg)
        sel, ign, strength, info = gws.step(t=1, candidates=[])
        self.assertIsNone(sel)
        self.assertFalse(ign)
        self.assertEqual(strength, 0.0)
        self.assertEqual(info.priorities.size, 0)
        self.assertEqual(gws.stats["steps"], 1)


class TestGWSAdvanced(unittest.TestCase):
    def test_soft_sampling_probabilities(self):
        cfg = GlobalWorkspaceConfig(content_dim=3, temperature=0.5, stochastic=True, seed=123)
        gws = GlobalWorkspace(cfg)
        cands = [
            GWSContent(vector=np.zeros(3), source="X", priority=1.0),
            GWSContent(vector=np.zeros(3), source="Y", priority=0.0),
        ]
        sel, ign, strength, info = gws.step(t=2, candidates=cands)
        # 概率存在且归一化
        self.assertIsNotNone(info.probs)
        self.assertAlmostEqual(float(np.sum(info.probs)), 1.0, places=6)
        self.assertGreaterEqual(info.probs[0], info.probs[1])

    def test_reset_stats(self):
        cfg = GlobalWorkspaceConfig(content_dim=2, ignition_threshold=0.1)
        gws = GlobalWorkspace(cfg)
        cands = [
            GWSContent(vector=np.zeros(2), source="S1", priority=0.2),
            GWSContent(vector=np.zeros(2), source="S2", priority=0.05),
        ]
        gws.step(t=0, candidates=cands)
        self.assertGreaterEqual(gws.stats["steps"], 1)
        self.assertGreaterEqual(gws.stats["ignitions"], 0)
        gws.reset_stats()
        self.assertEqual(gws.stats["steps"], 0)
        self.assertEqual(gws.stats["ignitions"], 0)
        self.assertEqual(gws.stats["per_source"], {})


if __name__ == "__main__":
    unittest.main()

