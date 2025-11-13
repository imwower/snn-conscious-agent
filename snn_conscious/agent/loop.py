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
from .policy_fast import FastPolicy, FastPolicyConfig
from .curiosity import Curiosity, CuriosityConfig
from snn_conscious.memory.episodic_memory import EpisodicMemory, EpisodicMemoryConfig
from snn_conscious.memory.replay import ReplayLearner, ReplayConfig

logger = logging.getLogger(__name__)


@dataclass
class LoopConfig:
    max_steps_per_episode: int = 200
    combine_intrinsic_in_reward: bool = True
    intrinsic_coef: float = 0.1
    replay_every_steps: int = 50
    replay_iters: int = 2
    replay_batch: int = 16


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
        self.replay = ReplayLearner(
            ReplayConfig(batch_size=loop_cfg.replay_batch, iters=loop_cfg.replay_iters),
            memory,
            wm,
        )

    def run_episode(self, seed: Optional[int] = None) -> dict:
        obs_raw = self.env.reset()
        obs_vec = self.encoder.encode(obs_raw)
        total_reward = 0.0
        intrinsic_total = 0.0
        for step in range(self.loop_cfg.max_steps_per_episode):
            # 策略选择动作
            a_id, a_onehot, pred_rew_choice = self.policy.act(obs_vec)
            # 世界模型提交一步（为了保持状态一致性）
            pred_obs, pred_rew, state_h = self.wm.step(obs_vec, a_onehot)
            # 环境交互
            next_obs_raw, ext_rew, done, info = self.env.step(a_id)
            next_obs_vec = self.encoder.encode(next_obs_raw)
            # 内在奖励
            intrinsic = self.curiosity.compute(pred_obs, next_obs_vec, pred_rew, ext_rew)
            # 组合奖励用于训练（可选）
            train_rew = ext_rew + (self.loop_cfg.intrinsic_coef * intrinsic if self.loop_cfg.combine_intrinsic_in_reward else 0.0)
            # 更新世界模型
            self.wm.update(pred_obs, next_obs_vec, pred_rew, train_rew)
            # 写入记忆
            self.memory.add((obs_vec, a_onehot, next_obs_vec, train_rew, done, step))

            total_reward += ext_rew
            intrinsic_total += intrinsic
            obs_vec = next_obs_vec

            if done:
                logger.info("[Loop] episode done at step=%d total_reward=%.3f intrinsic=%.3f", step + 1, total_reward, intrinsic_total)
                break

            if (step + 1) % self.loop_cfg.replay_every_steps == 0:
                self.replay.run_once()

        return {
            "steps": step + 1,
            "total_reward": total_reward,
            "intrinsic_total": intrinsic_total,
        }


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

