"""世界模型（SNN 近似原型）

该模块实现一个简化的“层级世界模型 SNN”原型，用于：
- 在离散时间步上预测下一步观测向量与奖励；
- 支持在线更新与离线重放（从 sample 学习）；
- 采用“脉冲近似”的内部状态：膜电位 v 与二值脉冲 s，更新规则使用简单离散时间公式。

设计说明（MVP 版本）：
- 模型结构：x(t)→隐藏层(膜电位 v, 脉冲 s, 低通 h)→线性读出→(pred_obs, pred_reward)
- 更新：
  - 隐层动力学：v[t] = leak * v[t-1] + W_in x[t] + W_rec s[t-1] + b_h
  - 触发脉冲：s[t] = H(v[t] - theta)，H 为阶跃（0/1），h[t] 为 s 的指数平滑（表示高层状态）。
  - 读出：pred_obs = W_ro_obs h + b_obs； pred_reward = W_ro_rew·h + b_rew
  - 学习：
    - 读出层使用 Delta Rule: ΔW_ro ∝ error × pre
    - 输入/循环权重使用三因子近似：ΔW_in/rec ∝ modulatory × (post × pre)，其中 modulatory ≈ |obs_error| 的均值 + λ_r * reward_error

未来扩展：
- 将离散脉冲近似替换为更真实的 LIF/AdEx 神经元；
- 使用 STDP/三因子更严格的形式；
- 支持多层层级 SNN 或变分世界模型。
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WorldModelConfig:
    """世界模型的超参数配置。

    属性：
        obs_dim: 观测向量维度（当前时刻的感知编码）。
        action_dim: 动作 one-hot 维度。
        context_dim: 上下文向量维度（可为 0）。
        hidden_dim: 隐藏层（SNN 神经元数量）。
        leak: 膜电位泄露系数（0~1）。
        threshold: 触发阈值（膜电位超过该值则产生脉冲）。
        lr_readout: 读出层学习率（Delta Rule）。
        lr_hidden: 隐层/输入/循环权重学习率（三因子近似）。
        reward_coef: 调制器中的奖励误差权重 λ_r。
        state_decay: 高层状态 h 的低通衰减系数（0~1）。
        clip_value: 权重裁剪范围（[-clip, clip]）。
        seed: 随机种子。
    """

    obs_dim: int
    action_dim: int
    context_dim: int = 0
    hidden_dim: int = 64
    leak: float = 0.95
    threshold: float = 1.0
    lr_readout: float = 0.05
    lr_hidden: float = 0.001
    reward_coef: float = 0.5
    state_decay: float = 0.9
    clip_value: float = 1.5
    seed: Optional[int] = None


class WorldModelSNN:
    """简化的层级世界模型（SNN 近似）。

    用法概述：
    - step(obs, action, context=None) → (pred_obs, pred_reward, state)
    - update(pred_obs, true_obs, pred_reward, true_reward) → 使用最近一次 step 的活动进行学习
    - learn_from_sample(...) → 离线重放接口：给定 sample，执行一次前向+学习

    说明：
    - 本实现以易读为先，内部均为 numpy 矩阵运算；未来可替换为 LIF+STDP。
    - 时间以离散步推进（self.t）。
    """

    def __init__(self, config: WorldModelConfig):
        if config.obs_dim <= 0 or config.action_dim <= 0 or config.hidden_dim <= 0:
            raise ValueError("obs_dim, action_dim, hidden_dim 必须为正整数")
        if config.context_dim < 0:
            raise ValueError("context_dim 不能为负")

        self.cfg = config
        self.rng = np.random.default_rng(config.seed)

        in_dim = config.obs_dim + config.action_dim + config.context_dim
        h_dim = config.hidden_dim

        # 权重初始化（小随机值）
        scale_in = 1.0 / max(1, in_dim)
        scale_h = 1.0 / max(1, h_dim)
        self.W_in = self.rng.normal(0.0, 0.5 * scale_in, size=(h_dim, in_dim))
        self.W_rec = self.rng.normal(0.0, 0.5 * scale_h, size=(h_dim, h_dim))
        self.b_h = np.zeros(h_dim, dtype=float)

        self.W_ro_obs = self.rng.normal(0.0, 0.5 * scale_h, size=(config.obs_dim, h_dim))
        self.b_obs = np.zeros(config.obs_dim, dtype=float)

        self.W_ro_rew = self.rng.normal(0.0, 0.5 * scale_h, size=(h_dim,))
        self.b_rew = 0.0

        # 内部状态：膜电位 v、脉冲 s、高层低通状态 h
        self.v = np.zeros(h_dim, dtype=float)
        self.s = np.zeros(h_dim, dtype=float)
        self.h = np.zeros(h_dim, dtype=float)

        # 记录上一次输入与状态，用于学习（eligibility 简化）
        self.last_input: Optional[np.ndarray] = None
        self.last_s: Optional[np.ndarray] = None
        self.t: int = 0

        logger.debug(
            "WorldModelSNN initialized: in_dim=%d, hidden_dim=%d", in_dim, h_dim
        )

    # --------------------------- 工具与校验 ---------------------------
    def _concat_input(
        self, obs: np.ndarray, action: np.ndarray, context: Optional[np.ndarray]
    ) -> np.ndarray:
        """拼接输入向量并做基本校验。"""
        if obs.shape != (self.cfg.obs_dim,):
            raise ValueError(f"obs 维度应为 ({self.cfg.obs_dim},), 实际为 {obs.shape}")
        if action.shape != (self.cfg.action_dim,):
            raise ValueError(
                f"action 维度应为 ({self.cfg.action_dim},), 实际为 {action.shape}"
            )
        if self.cfg.context_dim == 0:
            if context is not None and context.size != 0:
                raise ValueError("该配置不需要 context，但传入了非空向量")
            context_vec = np.zeros(0, dtype=float)
        else:
            if context is None:
                raise ValueError("该配置需要 context，但未传入")
            if context.shape != (self.cfg.context_dim,):
                raise ValueError(
                    f"context 维度应为 ({self.cfg.context_dim},), 实际为 {context.shape}"
                )
            context_vec = context
        return np.concatenate([obs, action, context_vec]).astype(float)

    @staticmethod
    def _heaviside(x: np.ndarray) -> np.ndarray:
        """阶跃函数 H(x)：x>0 → 1，否则 0。输出为 {0,1} 的 float。"""
        return (x > 0).astype(float)

    def _clip_weights(self) -> None:
        c = self.cfg.clip_value
        if c is None or c <= 0:
            return
        np.clip(self.W_in, -c, c, out=self.W_in)
        np.clip(self.W_rec, -c, c, out=self.W_rec)
        np.clip(self.W_ro_obs, -c, c, out=self.W_ro_obs)
        np.clip(self.W_ro_rew, -c, c, out=self.W_ro_rew)

    # --------------------------- 前向模拟 ---------------------------
    def step(
        self, obs: np.ndarray, action: np.ndarray, context: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """执行一步前向仿真，返回预测结果。

        参数：
            obs: 当前观测编码向量（长度 obs_dim）。
            action: 当前动作 one-hot（长度 action_dim）。
            context: 可选的上下文向量（长度 context_dim）。

        返回：
            pred_obs: 对下一步观测的预测（实数向量，长度 obs_dim）。
            pred_reward: 对下一步奖励的预测（标量）。
            state_repr: 高层世界状态表示 h（长度 hidden_dim）。
        """
        x = self._concat_input(obs, action, context)

        # 隐层动力学（离散时间）
        v_prev = self.v
        s_prev = self.s
        self.v = (
            self.cfg.leak * self.v
            + self.W_in @ x
            + self.W_rec @ self.s
            + self.b_h
        )
        self.s = self._heaviside(self.v - self.cfg.threshold)
        # 低通的“高层状态表示”
        self.h = self.cfg.state_decay * self.h + (1.0 - self.cfg.state_decay) * self.s

        # 读出层
        pred_obs = self.W_ro_obs @ self.h + self.b_obs
        pred_reward = float(np.dot(self.W_ro_rew, self.h) + self.b_rew)

        # 记录用于学习的活动痕迹
        self.last_input = x
        self.last_s = s_prev.copy()
        self.t += 1

        # 日志（简短，避免过多 IO）
        if self.t % 10 == 1:  # 降低日志频率
            spike_rate = float(np.mean(self.s))
            logger.info(
                "[WorldModel.step] t=%d spike_rate=%.3f pred_reward=%.4f",
                self.t,
                spike_rate,
                pred_reward,
            )

        return pred_obs, pred_reward, self.h.copy()

    # --------------------------- 学习（在线/离线） ---------------------------
    def update(
        self,
        pred_obs: np.ndarray,
        true_obs: np.ndarray,
        pred_reward: float,
        true_reward: float,
    ) -> None:
        """使用最近一次 step 的活动进行权重更新。

        更新规则：
        - 读出层：Delta Rule（最小二乘近似）
            ΔW_ro_obs = lr * (true_obs - pred_obs) ⊗ h
            ΔW_ro_rew = lr * (true_reward - pred_reward) * h
        - 隐层/输入/循环：简单三因子近似，调制器 m ≈ mean(|obs_err|) + λ_r * rew_err
            ΔW_in  ∝ m * (s ⊗ x)
            ΔW_rec ∝ m * (s ⊗ s_prev)
        """
        if self.last_input is None:
            logger.warning("update 被调用但缺少最近一次活动痕迹（请先执行 step）")
            return

        if true_obs.shape != (self.cfg.obs_dim,):
            raise ValueError("true_obs 维度不匹配")

        # 误差
        e_obs = true_obs.astype(float) - pred_obs.astype(float)
        e_rew = float(true_reward) - float(pred_reward)

        # 读出层 Delta Rule
        # W_ro_obs: (obs_dim, hidden_dim), h: (hidden_dim,)
        self.W_ro_obs += self.cfg.lr_readout * np.outer(e_obs, self.h)
        self.b_obs += self.cfg.lr_readout * e_obs

        # 奖励读出（标量）
        self.W_ro_rew += self.cfg.lr_readout * e_rew * self.h
        self.b_rew += self.cfg.lr_readout * e_rew

        # 调制器（可视作“全局 neuromodulator”）
        m = float(np.mean(np.abs(e_obs)) + self.cfg.reward_coef * e_rew)

        # 三因子近似（若 last_s 为空，则视作零向量）
        s_prev = self.last_s if self.last_s is not None else np.zeros_like(self.s)
        self.W_in += self.cfg.lr_hidden * m * np.outer(self.s, self.last_input)
        self.b_h += self.cfg.lr_hidden * m * self.s
        self.W_rec += self.cfg.lr_hidden * m * np.outer(self.s, s_prev)

        # 数值安全：裁剪权重
        self._clip_weights()

        if self.t % 10 == 1:
            logger.info(
                "[WorldModel.update] t=%d |obs_err|=%.4f rew_err=%.4f m=%.4f",
                self.t,
                float(np.mean(np.abs(e_obs))),
                float(e_rew),
                m,
            )

    # 便捷的离线重放学习接口
    def learn_from_sample(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        next_obs: np.ndarray,
        reward: float,
        context: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        """使用记忆样本执行一次前向+学习（可在重放阶段调用）。

        返回：(pred_obs, pred_reward)，便于外部记录学习曲线。
        """
        pred_obs, pred_rew, _ = self.step(obs, action, context)
        self.update(pred_obs, next_obs, pred_rew, reward)
        return pred_obs, pred_rew

    # 状态复位（例如 episode 之间）
    def reset_state(self) -> None:
        self.v.fill(0.0)
        self.s.fill(0.0)
        self.h.fill(0.0)
        self.last_input = None
        self.last_s = None
        self.t = 0
        logger.info("[WorldModel.reset_state] 状态已复位")


__all__ = ["WorldModelConfig", "WorldModelSNN"]

