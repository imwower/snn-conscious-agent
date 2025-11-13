"""Agent 主循环（在线 + 离线重放）MVP。

流程：
- 环境 reset，Perception 编码；
- 策略基于世界模型 predict 选择动作；
- 世界模型 step（提交选择），得到预测；
- 与环境交互得到 next_obs 与 extrinsic reward；
- 好奇心计算 intrinsic（可用于日志与可选组合到训练奖励中）；
- 调用世界模型 update；写入记忆；
- 若需要，执行离线 Replay。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import logging
import numpy as np

from snn_conscious.envs.gridworld import GridWorld, GridConfig
from snn_conscious.core.perception import PerceptionEncoder, PerceptionConfig
from snn_conscious.core.world_model import WorldModelSNN, WorldModelConfig
from snn_conscious.core.gws import GlobalWorkspace, GlobalWorkspaceConfig, GWSContent
from snn_conscious.memory.working_memory import WorkingMemory, WorkingMemoryConfig, WMItem
from snn_conscious.meta.self_model import SelfModel, SelfModelConfig
from .policy_fast import FastPolicy, FastPolicyConfig
from .policy_slow import SlowPolicy, SlowPolicyConfig
from .curiosity import Curiosity, CuriosityConfig
from snn_conscious.memory.episodic_memory import EpisodicMemory, EpisodicMemoryConfig
from snn_conscious.memory.replay import ReplayLearner, ReplayConfig
from snn_conscious.utils.stats import StatsRecorder

logger = logging.getLogger(__name__)


@dataclass
class LoopConfig:
    max_steps_per_episode: int = 200
    combine_intrinsic_in_reward: bool = True
    intrinsic_coef: float = 0.1
    # 策略切换（自我模型）
    use_slow_policy: bool = True
    think_threshold: float = 0.6
    slow_horizon: int = 2
    slow_gamma: float = 0.9
    replay_every_steps: int = 50
    replay_iters: int = 2
    replay_batch: int = 16
    # GWS 反馈
    gws_replay_on_ignition: bool = True
    gws_intrinsic_boost: float = 1.5
    # 统计输出
    stats_csv: Optional[str] = None


class AgentLoop:
    def __init__(
        self,
        env: GridWorld,
        encoder: PerceptionEncoder,
        wm: WorldModelSNN,
        policy: FastPolicy,
        curiosity: Curiosity,
        memory: EpisodicMemory,
        loop_cfg: LoopConfig,
    ):
        self.env = env
        self.encoder = encoder
        self.wm = wm
        self.policy = policy
        self.curiosity = curiosity
        self.memory = memory
        self.loop_cfg = loop_cfg
        # 全局工作区、工作记忆、自我模型
        self.gws = GlobalWorkspace(
            GlobalWorkspaceConfig(
                content_dim=self.wm.hidden_size,
                ignition_threshold=0.5,
                temperature=1.0,
                stochastic=False,
                topk_log_k=0,
            )
        )
        self.working_memory = WorkingMemory(
            WorkingMemoryConfig(content_dim=self.wm.hidden_size, capacity=32, decay=0.9)
        )
        self.self_model = SelfModel(
            SelfModelConfig(
                input_dim=self.wm.hidden_size + 2,
                lr=0.05,
                think_threshold=self.loop_cfg.think_threshold,
            )
        )

        self.replay = ReplayLearner(
            ReplayConfig(batch_size=loop_cfg.replay_batch, iters=loop_cfg.replay_iters),
            memory,
            wm,
        )
        self.slow_policy = SlowPolicy(
            SlowPolicyConfig(
                action_dim=GridWorld.action_dim(),
                gamma=self.loop_cfg.slow_gamma,
                horizon=self.loop_cfg.slow_horizon,
            ),
            wm,
        )
        self.stats = StatsRecorder()
        self._prev_obs_err = 0.0
        self._prev_rew_err = 0.0
        self._next_intrinsic_boost = 1.0

    def run_episode(self, seed: Optional[int] = None) -> dict:
        obs_raw = self.env.reset()
        obs_vec = self.encoder.encode(obs_raw)
        total_reward = 0.0
        intrinsic_total = 0.0
        for step in range(self.loop_cfg.max_steps_per_episode):
            # 自我模型评估：基于上一时刻隐状态与误差特征
            sm_x_pre = np.concatenate(
                [self.wm.h.copy(), np.array([self._prev_obs_err, self._prev_rew_err], dtype=float)]
            )
            conf_pre, risk_pre, should = self.self_model.estimate(sm_x_pre)

            # 策略选择动作（should 时切换慢策略）
            if self.loop_cfg.use_slow_policy and should:
                a_id, a_onehot, pred_rew_choice = self.slow_policy.act(obs_vec)
                policy_used = "slow"
            else:
                a_id, a_onehot, pred_rew_choice = self.policy.act(obs_vec)
                policy_used = "fast"
            # 世界模型提交一步（为了保持状态一致性）
            pred_obs, pred_rew, state_h = self.wm.step(obs_vec, a_onehot)
            # 环境交互
            next_obs_raw, ext_rew, done, info = self.env.step(a_id)
            next_obs_vec = self.encoder.encode(next_obs_raw)
            # 内在奖励与误差特征
            intrinsic = self.curiosity.compute(pred_obs, next_obs_vec, pred_rew, ext_rew)
            obs_err = float(np.mean(np.abs(next_obs_vec - pred_obs)))
            rew_err = abs(float(ext_rew) - float(pred_rew))
            # 组合奖励用于训练（可选），受上一时刻 GWS 的 boost 影响
            intrinsic_coef_step = self.loop_cfg.intrinsic_coef * self._next_intrinsic_boost
            train_rew = ext_rew + (
                intrinsic_coef_step * intrinsic if self.loop_cfg.combine_intrinsic_in_reward else 0.0
            )
            # 更新世界模型
            self.wm.update(pred_obs, next_obs_vec, pred_rew, train_rew)
            # 更新自我模型（使用正外在奖励作为成功信号的简化近似）
            sm_x_post = np.concatenate([state_h, np.array([obs_err, rew_err], dtype=float)])
            self.self_model.update(sm_x_post, target_success=int(ext_rew > 0.0))

            # 写入记忆
            self.memory.add((obs_vec, a_onehot, next_obs_vec, train_rew, done, step))

            # 构建 GWS 候选并竞争广播
            candidates = [
                GWSContent(vector=state_h, source="world_model", priority=float(intrinsic)),
            ]
            # 工作记忆聚合表示作为候选（低优先级）
            wm_vec = self.working_memory.read_vector()
            if np.any(wm_vec):
                candidates.append(
                    GWSContent(vector=wm_vec, source="working_memory", priority=0.1)
                )
            # 自我模型内容（使用风险作为优先级）
            candidates.append(
                GWSContent(vector=state_h, source="self_model", priority=float(risk_pre))
            )

            selected, ign, strength, _info = self.gws.step(t=step, candidates=candidates)
            if ign and selected is not None:
                # 点火则将内容写入工作记忆
                self.working_memory.add(
                    WMItem(vector=selected.vector, source=selected.source, t=step, strength=1.0)
                )
                if self.loop_cfg.gws_replay_on_ignition:
                    self.replay.run_once()
                self._next_intrinsic_boost = self.loop_cfg.gws_intrinsic_boost
            else:
                self._next_intrinsic_boost = 1.0

            # 记录统计
            self.stats.add(
                {
                    "step": step + 1,
                    "policy": policy_used,
                    "ext_rew": float(ext_rew),
                    "intrinsic": float(intrinsic),
                    "pred_rew": float(pred_rew),
                    "obs_err": obs_err,
                    "rew_err": rew_err,
                    "sm_conf": float(conf_pre),
                    "sm_risk": float(risk_pre),
                    "gws_ign": int(ign),
                    "gws_src": selected.source if (selected is not None) else "",
                    "spike_rate": float(self.wm.spike_rate),
                }
            )

            total_reward += ext_rew
            intrinsic_total += intrinsic
            obs_vec = next_obs_vec
            self._prev_obs_err = obs_err
            self._prev_rew_err = rew_err

            if done:
                logger.info("[Loop] episode done at step=%d total_reward=%.3f intrinsic=%.3f", step + 1, total_reward, intrinsic_total)
                break

            if (step + 1) % self.loop_cfg.replay_every_steps == 0:
                self.replay.run_once()

        result = {
            "steps": step + 1,
            "total_reward": total_reward,
            "intrinsic_total": intrinsic_total,
        }
        if self.loop_cfg.stats_csv:
            self.stats.to_csv(self.loop_cfg.stats_csv)
        return result


def build_mvp(seed: Optional[int] = 0):
    # 组件装配（默认 5x5 网格）
    grid_cfg = GridConfig(seed=seed)
    env = GridWorld(grid_cfg)

    enc_cfg = PerceptionConfig(width=grid_cfg.width, height=grid_cfg.height)
    encoder = PerceptionEncoder(enc_cfg)

    wm_cfg = WorldModelConfig(
        obs_dim=enc_cfg.obs_dim(),
        action_dim=GridWorld.action_dim(),
        hidden_dim=64,
        seed=seed,
    )
    wm = WorldModelSNN(wm_cfg)

    pol_cfg = FastPolicyConfig(action_dim=GridWorld.action_dim(), epsilon=0.1)
    policy = FastPolicy(pol_cfg, wm)

    cur_cfg = CuriosityConfig(alpha=1.0, beta=0.5, normalize=False)
    curiosity = Curiosity(cur_cfg)

    mem = EpisodicMemory(EpisodicMemoryConfig(capacity=5000, seed=seed))
    loop_cfg = LoopConfig()

    loop = AgentLoop(env, encoder, wm, policy, curiosity, mem, loop_cfg)
    return loop


__all__ = ["LoopConfig", "AgentLoop", "build_mvp"]
