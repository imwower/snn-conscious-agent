"""简单的时间序列统计记录器。

用于在训练/运行过程中记录每步的指标，并可保存为 CSV。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any
import csv
import os


@dataclass
class StatsRecorder:
    keys: List[str] = field(default_factory=list)
    rows: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, row: Dict[str, Any]) -> None:
        if not self.keys:
            self.keys = list(row.keys())
        self.rows.append(row)

    def to_csv(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if not self.rows:
            # 空数据也创建文件，便于流水线
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.keys or [])
            return
        # 统一列顺序
        fieldnames = self.keys
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.rows:
                writer.writerow({k: r.get(k, "") for k in fieldnames})


__all__ = ["StatsRecorder"]

