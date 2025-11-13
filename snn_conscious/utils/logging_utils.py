"""日志配置工具。

在 CLI 或脚本中可调用 `setup_basic_logging()` 以获得统一风格的日志输出。
"""

from __future__ import annotations

import logging


def setup_basic_logging(level: int = logging.INFO) -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )


__all__ = ["setup_basic_logging"]

