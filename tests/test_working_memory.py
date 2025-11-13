import unittest
import numpy as np

from snn_conscious.memory.working_memory import WorkingMemory, WorkingMemoryConfig, WMItem


class TestWorkingMemory(unittest.TestCase):
    def test_add_and_read(self):
        cfg = WorkingMemoryConfig(content_dim=5, capacity=3, decay=0.5)
        wm = WorkingMemory(cfg)
        v1 = np.array([1, 0, 0, 0, 0], dtype=float)
        v2 = np.array([0, 1, 0, 0, 0], dtype=float)
        wm.add(WMItem(vector=v1, source="s1", t=0))
        wm.add(WMItem(vector=v2, source="s2", t=1))
        vec = wm.read_vector()
        self.assertEqual(vec.shape, (cfg.content_dim,))
        self.assertGreater(vec[0], 0.0)
        self.assertGreater(vec[1], 0.0)

    def test_capacity_and_reset(self):
        cfg = WorkingMemoryConfig(content_dim=3, capacity=2, decay=0.9)
        wm = WorkingMemory(cfg)
        wm.add(WMItem(vector=np.ones(3), source="a", t=0))
        wm.add(WMItem(vector=np.ones(3), source="b", t=1))
        wm.add(WMItem(vector=np.ones(3), source="c", t=2))  # 触发淘汰
        self.assertEqual(len(wm.buffer), 2)
        wm.reset()
        self.assertEqual(len(wm.buffer), 0)


if __name__ == "__main__":
    unittest.main()

