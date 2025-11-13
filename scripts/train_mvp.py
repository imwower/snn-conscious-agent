from __future__ import annotations

import logging

from snn_conscious.utils.logging_utils import setup_basic_logging
from snn_conscious.agent.loop import build_mvp


def main():
    setup_basic_logging(logging.INFO)
    loop = build_mvp(seed=0)
    stats = loop.run_episode()
    print("Episode stats:", stats)


if __name__ == "__main__":
    main()

