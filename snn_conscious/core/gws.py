"""意识核 / 全局工作区（Global Workspace, GWS）原型实现。

职责：
- 接收来自多个子模块的候选内容（向量 + 源信息 + 优先级）；
- 基于竞争机制（argmax 或 softmax 采样）选择当前时间步的“意识内容”；
- 当被选中内容的优先级超过阈值时，产生一次“点火(ignition)”事件；
- 广播所选内容；记录统计信息与可选的 top-K 候选，用于后续分析。

本实现为 MVP，侧重清晰与可扩展：
- 使用 numpy 完成向量化；
- 离散时间步接口：外部传入 t=int；
- 提供重置统计接口 reset_stats；
- 预留 soft-WTA 温度与随机采样开关，便于后续研究。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Dict, Any
import logging

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GlobalWorkspaceConfig:
    """GWS 的超参数配置。

    属性：
        content_dim: 候选内容向量的固定维度。
        ignition_threshold: 点火阈值（优先级 >= 阈值则触发点火）。
        temperature: soft-WTA 温度（>0）。温度越小，分布越尖锐。
        stochastic: 若为 True，则使用 softmax 采样；否则使用 argmax 选择。
        topk_log_k: 若 >0，记录每步的 top-K 候选（源名与分数）。
        seed: 采样与微扰的随机种子（仅在 stochastic=True 或添加噪声时使用）。
    """

    content_dim: int
    ignition_threshold: float = 1.0
    temperature: float = 1.0
    stochastic: bool = False
    topk_log_k: int = 0
    seed: Optional[int] = None


@dataclass
class GWSContent:
    """候选内容对象。

    字段：
        vector: 内容向量（形状：(content_dim,) 的 numpy.ndarray）。
        source: 源模块名称（用于调试与统计）。
        priority: 优先级分数（实数，可由各模块根据误差/新奇度/价值等计算）。
    """

    vector: np.ndarray
    source: str
    priority: float


@dataclass
class GWSStepInfo:
    """一步选择的统计快照，用于分析与测试。"""

    t: int
    priorities: np.ndarray
    selected_idx: Optional[int]
    probs: Optional[np.ndarray] = None
    topk: Optional[List[Tuple[str, float]]] = None


class GlobalWorkspace:
    """意识核 / 全局工作区（GWS）

    核心接口：
        - step(t, candidates) -> (selected, ignition, ignition_strength, info)
        - reset_stats()

    说明：
        - 当 candidates 为空时，不选择任何内容，ignition=False。
        - ignition_strength 定义为 max(0, selected.priority - threshold)。
    """

    def __init__(self, config: GlobalWorkspaceConfig):
        if config.content_dim <= 0:
            raise ValueError("content_dim 必须为正整数")
        if config.temperature <= 0:
            raise ValueError("temperature 必须 > 0")
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)

        # 统计信息（可按需扩展）
        self.stats: Dict[str, Any] = {
            "steps": 0,
            "ignitions": 0,
            "selected_priority": [],
            "selected_source": [],
            "per_source": {},  # 计数：每个源被选中的次数
        }

        # 最近一次被广播的内容（便于其它模块读取）
        self.last_selected: Optional[GWSContent] = None
        self.last_info: Optional[GWSStepInfo] = None

        logger.debug(
            "GlobalWorkspace initialized: content_dim=%d, threshold=%.3f, temp=%.3f, stochastic=%s",
            self.cfg.content_dim,
            self.cfg.ignition_threshold,
            self.cfg.temperature,
            str(self.cfg.stochastic),
        )

    # ------------------------------ 工具方法 ------------------------------
    def _validate_candidates(self, candidates: Sequence[GWSContent]) -> None:
        for c in candidates:
            v = np.asarray(c.vector, dtype=float)
            if v.shape != (self.cfg.content_dim,):
                raise ValueError(
                    f"候选内容向量维度应为 ({self.cfg.content_dim},), 实际为 {v.shape}"
                )

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x_max = np.max(x) if x.size > 0 else 0.0
        ex = np.exp(x - x_max)
        s = ex.sum()
        return ex / s if s > 0 else np.zeros_like(x)

    # ------------------------------- 主流程 -------------------------------
    def step(
        self, t: int, candidates: Sequence[GWSContent]
    ) -> Tuple[Optional[GWSContent], bool, float, GWSStepInfo]:
        """执行一步竞争与广播。

        参数：
            t: 当前离散时间步。
            candidates: 候选内容列表（可为空）。

        返回：
            selected: 被选中的内容（可能为 None）。
            ignition: 是否触发点火事件。
            ignition_strength: 点火强度（>=0）。
            info: 本步统计快照（优先级列表、topK、概率等）。
        """
        if t < 0:
            raise ValueError("时间步 t 需为非负整数")

        if len(candidates) == 0:
            info = GWSStepInfo(t=t, priorities=np.zeros(0), selected_idx=None, probs=None, topk=[])
            self.last_selected = None
            self.last_info = info
            # 记录统计
            self.stats["steps"] += 1
            logger.debug("[GWS.step] t=%d 无候选内容", t)
            return None, False, 0.0, info

        # 校验维度
        self._validate_candidates(candidates)

        pri = np.array([float(c.priority) for c in candidates], dtype=float)

        # soft-WTA: 根据温度得到概率分布（仅在 stochastic=True 时采样）
        logits = pri / self.cfg.temperature
        probs = self._softmax(logits) if self.cfg.stochastic else None

        if self.cfg.stochastic:
            # 概率采样
            idx = int(self.rng.choice(len(candidates), p=probs))
        else:
            # 纯粹的 argmax 选择
            idx = int(np.argmax(pri))

        selected = candidates[idx]
        ignition_strength = max(0.0, float(selected.priority) - float(self.cfg.ignition_threshold))
        ignition = ignition_strength > 0.0

        # 统计与日志
        self.stats["steps"] += 1
        self.stats["selected_priority"].append(float(selected.priority))
        self.stats["selected_source"].append(selected.source)
        self.stats["per_source"][selected.source] = self.stats["per_source"].get(selected.source, 0) + 1
        if ignition:
            self.stats["ignitions"] += 1
            logger.info(
                "[GWS.ignition] t=%d source=%s priority=%.4f strength=%.4f",
                t,
                selected.source,
                float(selected.priority),
                ignition_strength,
            )

        # 记录 top-K（可选）
        topk_list: Optional[List[Tuple[str, float]]] = None
        if self.cfg.topk_log_k and self.cfg.topk_log_k > 0:
            order = np.argsort(-pri)
            k = min(self.cfg.topk_log_k, len(candidates))
            topk_list = [(candidates[i].source, float(pri[i])) for i in order[:k]]
            logger.debug("[GWS.topk] t=%d top%d=%s", t, k, topk_list)

        # 保存本步信息
        info = GWSStepInfo(t=t, priorities=pri, selected_idx=idx, probs=probs, topk=topk_list)
        self.last_selected = selected
        self.last_info = info

        return selected, ignition, ignition_strength, info

    # ----------------------------- 管理接口 -----------------------------
    def reset_stats(self) -> None:
        """重置统计信息（用于不同 episode 之间）。"""
        self.stats = {
            "steps": 0,
            "ignitions": 0,
            "selected_priority": [],
            "selected_source": [],
            "per_source": {},
        }
        self.last_selected = None
        self.last_info = None
        logger.info("[GWS.reset_stats] 统计信息已清理")


__all__ = [
    "GlobalWorkspaceConfig",
    "GWSContent",
    "GWSStepInfo",
    "GlobalWorkspace",
]

