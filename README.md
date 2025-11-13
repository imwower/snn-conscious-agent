# snn-conscious-agent

一个基于脉冲神经网络（Spiking Neural Network, SNN）的类脑智能体原型，用来探索：

- 🌐 全局工作区（Global Workspace / 意识核）
- 🔮 预测型世界模型（Predictive World Model）
- 🧍 自我模型 / 元认知（Self-Model & Metacognition）
- 🧠 多时间尺度记忆 + 重放（Working / Episodic Memory & Replay）
- 🔍 好奇心 / 内在驱动（Curiosity / Intrinsic Motivation）
- 🤖 具身环境中的自主行为（Embodied Agent）

> 本项目的目标不是“证明有真正主观体验”，而是工程化实现一套功能上接近自主意识的 SNN agent，并通过行为与内部动力学进行验证和实验。

---

## 目录

- 1. 项目目标
- 2. 整体架构
  - 2.1 模块说明（概览）
- 3. 目录结构
- 4. MVP 阶段设计
  - 4.1 MVP 功能目标
  - 4.2 MVP 后的扩展阶段
- 5. 依赖与运行
  - 5.1 环境依赖（建议）
  - 5.2 运行示例
- 6. 评估指标（建议）
- 7. TODO & 扩展方向

---

## 1. 项目目标

1. 构建一个具身 SNN 智能体  
   - 在简化网格世界/迷宫中完成导航、收集、解锁等任务。
   - 使用 SNN（离散时间 / 脉冲事件）实现感知、预测、策略。

2. 在 SNN 中实现“意识功能模块”  
   - 全局工作区（意识核）：重要信息的点火与广播。
   - 世界模型：持续预测下一步感觉和结果。
   - 自我模型：评估置信度/错误风险，决定“要不要多想一步”。
   - 记忆与重放：支持短期/情景记忆，并在休息期自动重放。
   - 好奇心：根据预测误差/新奇度驱动探索与“内部思考”。

3. 支持可视化和实验  
   - 记录各模块脉冲活动、误差信号、意识点火事件。
   - 方便扩展新模块或替换学习规则，进行对照实验。

---

## 2. 整体架构

高层模块结构如下：

```text
       ┌───────────── 环境 / 世界 ──────────────┐
       │                                        │
[感知编码 SNN] → [层级世界模型 SNN] ───┐        │
       │                               │        │
       │                        ┌─[好奇心/价值 SNN]─┐
       │                        │        ↑        │
       ↓                        ↓        │        │
[工作记忆/情景记忆 SNN] → [意识核 / 全局工作区 SNN] ← [自我模型/元认知 SNN]
       ↑                        ↓
       │                  [任务/行为策略 SNN]
       │                        ↓
       └─────────────── [身体 / 执行器] ────────────┘
```

### 2.1 模块说明（概览）

- `core/perception.py`：感知编码 SNN。将环境状态编码为脉冲时间模式。
- `core/world_model.py`：层级世界模型 SNN。学习 p(s_{t+1} | s_t, a_t)，并在无外部输入时进行内部模拟。
- `core/gws.py`：意识核 / 全局工作区 SNN。实现内容竞争、点火（ignition）与全局广播。
- `meta/self_model.py`：自我模型 / 元认知 SNN。估计当前置信度、错误风险，“知道自己知道/不知道”。
- `memory/working_memory.py`：工作记忆 SNN。短时间维护意识内容和关键任务信息。
- `memory/episodic_memory.py`：情景记忆 SNN。把 (state, action, reward) 序列编码成可重放的轨迹。
- `memory/replay.py`：记忆重放控制。在“休息”阶段选择高价值/高误差轨迹进行重放和再学习。
- `agent/curiosity.py`：好奇心 / 价值系统。把预测误差、新奇度、奖励整合为内在动机信号。
- `agent/policy_fast.py`、`agent/policy_slow.py`：行为策略 SNN。快通路（熟练反射）、慢通路（调动记忆与推理）。
- `agent/loop.py`：主循环。串联感知 → 预测 → 意识核 → 自我模型 → 策略 → 环境交互 → 记忆。
- `envs/gridworld.py`：简化网格环境。MVP 阶段的测试环境（迷宫、钥匙-门、小奖励等）。

---

## 3. 目录结构

建议的仓库目录结构如下：

```text
snn-conscious-agent/
├── README.md
├── pyproject.toml        # 或 setup.cfg / requirements.txt（依赖管理）
├── snn_conscious/
│   ├── __init__.py
│   ├── config.py         # 全局配置（时间步长、神经元参数等）
│   ├── core/
│   │   ├── perception.py # 感知编码 SNN
│   │   ├── world_model.py# 预测型世界模型 SNN
│   │   └── gws.py        # 意识核 / 全局工作区 SNN
│   ├── meta/
│   │   ├── self_model.py # 自我模型 / 元认知 SNN
│   │   └── introspection.py # （可选）内部状态分析工具
│   ├── memory/
│   │   ├── working_memory.py  # 短期/工作记忆
│   │   ├── episodic_memory.py # 情景记忆
│   │   └── replay.py          # 重放调度逻辑
│   ├── agent/
│   │   ├── policy_fast.py # 快速策略 SNN
│   │   ├── policy_slow.py # 慢速策略 SNN
│   │   ├── curiosity.py   # 好奇心 / 内在价值系统
│   │   └── loop.py        # 主循环（在线 / 离线）
│   ├── envs/
│   │   └── gridworld.py   # 简化网格环境
│   └── utils/
│       ├── logging_utils.py   # 日志/可视化辅助
│       ├── snn_layers.py      # 通用 SNN 层/神经元/学习规则
│       └── serialization.py   # 模型保存/加载
├── experiments/
│   ├── exp_mvp_gridworld.py     # MVP：最小可运行示例
│   └── exp_ablation_no_gws.py   # 消融实验（无意识核）
├── tests/
│   ├── test_world_model.py
│   ├── test_gws.py
│   ├── test_self_model.py
│   └── test_memory_replay.py
└── scripts/
    ├── run_mvp.sh          # 方便一键跑 MVP
    └── train_mvp.py        # 训练脚本入口
```

---

## 4. MVP 阶段设计（最小可运行版本）

### 4.1 MVP 功能目标

环境：简化 GridWorld（二维网格），agent 需要：

- 找到目标格子（奖励点）；
- 避开障碍；
- 完成简单多步任务（例如：先拿钥匙再开门）。

模块最小子集：

- `core/perception.py`：编码 agent 的位置 + 邻近格子信息；
- `core/world_model.py`：预测下一步观察/奖励；
- `agent/policy_fast.py`：基于世界模型输出动作；
- `memory/episodic_memory.py` + `memory/replay.py`：简单重放；
- `agent/curiosity.py`：用预测误差作为内在奖励；
- `agent/loop.py`：在线循环 + 简单离线重放。

训练与评估：

- 度量能否在合理步数内完成任务；
- 比较有/无重放时的学习速度差异。

### 4.2 MVP 后的扩展阶段

- Phase 2：加入 意识核 / GWS
- Phase 3：加入 自我模型 / 不确定性估计
- Phase 4：加入 多时间尺度记忆 + 高级重放策略
- Phase 5：加入 内部问题生成 + 语言/DSL 桥接层（可选）

README 中可以逐步更新每个阶段的状态与 TODO。

---

## 5. 依赖与运行

### 5.1 环境依赖（建议）

- Python 3.10+
- 标准库：`logging`, `dataclasses`, `typing`, `math`, `random`, `itertools`, `json` 等
- 推荐但非必须：
  - `numpy`：高效向量运算（可以先支持纯 Python，再加速）
  - `matplotlib`：可选的可视化
  - （可选）`gymnasium`：若需要接入标准环境接口

### 5.2 运行示例

```bash
# 安装依赖（开发模式）
pip install -e .

# 运行最小示例
python scripts/train_mvp.py

# 或使用 shell 脚本
bash scripts/run_mvp.sh
```

---

## 6. 评估指标（建议）

任务表现：

- 平均成功率、平均步数、平均奖励。

学习效率：

- 不同阶段的学习曲线（有/无意识核、有/无重放的对比）。

内部指标：

- 世界模型预测误差随时间变化；
- 意识核点火频率、点火时涉及的模块；
- 自我模型置信度 vs 实际成功率的一致性（元认知校准）。

“类意识”行为（定性观察）：

- 在不确定时延长决策时间 / 调动更多模块；
- 在休息阶段出现有结构的内部重放活动；
- 在某些情景上表现出“谨慎 / 探索 / 纠错”的稳定策略。

---

## 7. TODO & 扩展方向

- 使用更真实的 SNN 神经元模型（如 LIF, AdEx），并支持可插拔的学习规则（STDP/三因子）。
- 增加不同类型环境（部分可观测、连续空间）。
- 加入简单语言/DSL 接口，让 agent 可以“说出”自己的意识内容与问题。
- 引入多 agent 交互，探索“多体意识”/共享工作区的行为。

