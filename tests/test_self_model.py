import unittest
import numpy as np

from snn_conscious.meta.self_model import SelfModel, SelfModelConfig


class TestSelfModel(unittest.TestCase):
    def test_estimate_ranges(self):
        cfg = SelfModelConfig(input_dim=6, seed=0)
        sm = SelfModel(cfg)
        x = np.zeros(cfg.input_dim)
        conf, risk, should = sm.estimate(x)
        self.assertTrue(0.0 < conf < 1.0)
        self.assertAlmostEqual(conf + risk, 1.0, places=6)

    def test_update_improves_separation(self):
        # 构造线性可分数据：前3维和为正 -> y=1，否则 y=0
        cfg = SelfModelConfig(input_dim=6, lr=0.1, seed=0)
        sm = SelfModel(cfg)
        rng = np.random.default_rng(0)

        # 训练
        for _ in range(200):
            x = rng.normal(0, 1, size=(cfg.input_dim,))
            y = 1 if np.sum(x[:3]) > 0 else 0
            sm.update(x, y)

        # 评估：随机抽样，正确率应 > 0.7
        correct = 0
        total = 200
        for _ in range(total):
            x = rng.normal(0, 1, size=(cfg.input_dim,))
            y = 1 if np.sum(x[:3]) > 0 else 0
            conf, _, _ = sm.estimate(x)
            pred = 1 if conf >= 0.5 else 0
            correct += int(pred == y)
        acc = correct / total
        self.assertGreater(acc, 0.7)


if __name__ == "__main__":
    unittest.main()

