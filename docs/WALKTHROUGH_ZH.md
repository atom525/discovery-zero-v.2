# Discovery Zero 研究流程完整指南

本文档以 **同手性机制（Homochirality Mechanism）** 这一真实案例为范例，完整展示如何使用 Discovery Zero 从零开始跑通一个前沿开放问题的探索流程。文中所有实验结果均为系统真实运行产出，未经手动编辑。

---

## 目录

1. [概念概述](#1-概念概述)
2. [环境准备](#2-环境准备)
3. [理解 Case 配置](#3-理解-case-配置)
4. [运行探索](#4-运行探索)
5. [理解输出产物](#5-理解输出产物)
6. [分析真实实验记录](#6-分析真实实验记录homochirality-showcase)
7. [定义你自己的研究问题](#7-定义你自己的研究问题)
8. [常见问题与排错](#8-常见问题与排错)

---

## 1. 概念概述

Discovery Zero 的核心理念：

```
开放问题 → 种子知识图谱 → MCTS 迭代探索 → 信念传播驱动 → 科学发现
```

系统的每一轮探索包括以下模块的协同工作：

| 阶段 | 模块 | 作用 |
|------|------|------|
| Plausible Reasoning | `orchestrator.run_plausible_action` | LLM 对目标命题进行推测性论证 |
| Claim Extraction | `ClaimPipeline` | 从论证文本中提取可验证声明 |
| Claim Verification | `ClaimVerifier` | 通过 Python 数值实验和/或 Lean 形式验证各声明 |
| Belief Propagation | `run_inference_v2`（内置 gaia_bp 引擎） | 根据验证结果更新全图节点置信度 |
| Judge | orchestrator 内置 | 评估论证质量，给出 Grade (A/B/C/D) |
| Bridge Planning | `BridgePlan` | 构建从种子到目标的推理路径 |
| MCTS | `MCTSDiscoveryEngine` | 用 UCB/RMaxTS 策略选择下一步动作 |

每次迭代后，系统检查目标节点的 belief 是否达到阈值（默认 0.95）。若未达到，MCTS 继续选择最有前景的方向探索。

---

## 2. 环境准备

### 2.1 安装项目

```bash
cd /personal/Discovery-Zero-v2
pip install -e ".[dev]"
```

### 2.2 配置 Python 路径

仓库已内置所有 BP/推理依赖（`src/gaia_bp/` + `libs/inference_v2/`），无需额外 Gaia 仓库：

```bash
export PYTHONPATH="$(pwd)/src:$(pwd)"
```

验证导入：

```bash
python -c "from discovery_zero.benchmark import run_suite; print('OK')"
```

### 2.3 配置 LLM

```bash
cp .env.example .env
# 编辑 .env，填入真实的 API 密钥
```

必填项：

| 变量 | 说明 |
|------|------|
| `LITELLM_PROXY_API_BASE` | LiteLLM 兼容网关地址 |
| `LITELLM_PROXY_API_KEY` | API 密钥 |
| `DISCOVERY_ZERO_LLM_MODEL` | 模型标识，如 `cds/Claude-4.6-opus` |

可选项：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DISCOVERY_ZERO_ENABLE_MCTS` | `1` | 是否启用 MCTS 探索 |
| `DISCOVERY_ZERO_MCTS_MAX_TIME_SECONDS` | `21600` | MCTS 最长运行时间（秒） |

### 2.4 Lean 环境（可选）

若需要形式验证功能：

```bash
make lean-ready     # 一键安装 elan + Lean 4 + Mathlib
make lean-verify    # 验证 lean_workspace 可编译
```

**注意**：同手性案例主要依赖 Python 数值实验验证，Lean 部分为可选增强。

---

## 3. 理解 Case 配置

每个研究案例由两个 JSON 文件定义：

### 3.1 `case.json` — 案例元信息

位于 `evaluate/cases/<case_id>/case.json`。示例（同手性）：

```json
{
  "case_id": "homochirality_mechanism",
  "display_name": "前生物同手性起源机制",
  "source_proof_config": "proof_config.json",
  "benchmark_scope": "Construct a complete prebiotic mechanism ...",
  "tags": ["chemistry", "prebiotic", "homochirality"],
  "model": "cds/Claude-4.6-opus",
  "timeouts": {
    "experiment": 180,
    "decompose": 240,
    "lean": 360
  },
  "lean_policy": {
    "mode": "selective",
    "enable_decomposition": false
  },
  "planning_constraints": [
    "The Frank model requires ee_0 != 0 as input ...",
    "Do not assume any single known mechanism is sufficient ..."
  ]
}
```

关键字段：

- **`benchmark_scope`**：问题的精确描述，系统据此生成目标节点
- **`planning_constraints`**：对系统推理的约束条件（如"不要假设 CPL 足够"）
- **`lean_policy`**：Lean 验证策略，非数学问题一般设 `mode: selective`
- **`timeouts`**：各阶段的超时限制（秒）

### 3.2 `proof_config.json` — 知识图谱种子

这是系统探索的起点，包含：

- **`seed_nodes`**：已知事实节点（belief = 1.0, state = "proven"）
- **`target`**：目标命题（初始 belief = 0.1, state = "unverified"）
- **`planning_constraints`**：图级别的推理约束

同手性案例的 4 个种子节点：

| 种子 | 内容 |
|------|------|
| `ee_and_homochirality_fact` | ee 的定义、生物同手性事实、Miller-Urey 产物外消旋 |
| `frank_model` | Frank 自催化放大模型的动力学方程 |
| `frank_model_symmetry_barrier` | Frank 模型的对称性壁垒：ee(0)=0 时无法打破 |
| `stochastic_and_prebiotic_constraints` | 随机涨落尺度 1/√N、前生物环境约束 |

目标节点：构建一个从 ee=0 出发、在早期地球条件下达到 |ee| > 0.999 的完整前生物机制。

### 3.3 `suite.json` — 测试套件

将多个 case 组合为一次批量运行：

```json
{
  "suite_id": "frontier_open_problems_v1",
  "cases": [
    "cases/lonely_runner_n11/case.json",
    "cases/homochirality_mechanism/case.json",
    ...
  ]
}
```

---

## 4. 运行探索

### 4.1 运行单个案例（推荐入门）

最简方式——创建一个只包含一个 case 的临时 suite：

```bash
# 创建单 case suite
cat > /tmp/single_case_suite.json << 'EOF'
{
  "suite_id": "homochirality_single",
  "display_name": "单案例：同手性机制",
  "repeats": 1,
  "run_mode": "serial",
  "cases": [
    "cases/homochirality_mechanism/case.json"
  ]
}
EOF

# 运行
source .env
export PYTHONPATH="$(pwd)/src:$(pwd)"

python scripts/run_benchmark_suite.py \
  --suite /tmp/single_case_suite.json \
  --output-root evaluate/workspaces
```

### 4.2 运行完整套件

```bash
source .env
export PYTHONPATH="$(pwd)/src:$(pwd)"

python scripts/run_benchmark_suite.py \
  --suite evaluate/suite.json \
  --output-root evaluate/workspaces
```

### 4.3 运行时间预估

| 参数 | 典型值 |
|------|--------|
| 单个 MCTS 迭代 | 2–5 分钟（取决于 LLM 延迟） |
| 同手性案例完整运行 | 约 3–6 小时（46 次迭代） |
| 完整 5-case 套件 | 约 15–30 小时（串行模式） |

可通过 `DISCOVERY_ZERO_MCTS_MAX_TIME_SECONDS` 控制最长时间。

---

## 5. 理解输出产物

运行结束后，输出位于 `evaluate/workspaces/runs/<suite_id>/<timestamp>/<case_id>/run_01/`。

### 5.1 文件结构

```
run_01/
├── summary.json              # 运行摘要：最终 belief、指标、状态
├── graph.json                # 最终知识超图快照
├── bridge_plan.json          # Bridge 推理路径
├── exploration_log.json      # MCTS 完整迭代记录
├── resolved_proof_config.json# 实际使用的种子配置
├── PATH_BENCHMARK_SCORECARD_ZH.md  # 可读的评分卡
└── llm_records/              # 所有 LLM 交互的完整记录
    ├── plausible_prose_attempt_1.txt   # 第 1 轮推测论证
    ├── plausible_prose_attempt_2.txt   # 第 2 轮（自我纠正后）
    ├── plausible_prose_attempt_3.txt   # 第 3 轮（最终机制）
    ├── bridge_plan_prose_attempt_*.txt # Bridge 规划文本
    ├── claim_extraction_attempt_1.txt  # 提取的可验证声明
    ├── claim_verify_quant_1_code.py    # 定量验证代码
    ├── claim_verify_quant_1_result.txt # 验证运行结果
    ├── claim_verify_struct_1_code.lean # 结构验证 Lean 代码
    ├── experiment_code_attempt_1.txt   # 数值实验代码
    ├── experiment_prose_attempt_1.txt  # 实验设计描述
    ├── experiment_result_attempt_1.txt # 实验运行输出
    └── lean_gap_analysis_*.txt         # Lean 形式化差距分析
```

### 5.2 关键指标（summary.json）

| 指标 | 含义 | 同手性案例值 |
|------|------|-------------|
| `final_target_belief` | 目标命题的最终置信度 | 0.815 |
| `benchmark_outcome` | 运行结果状态 | `bridge_consumption_ready` |
| `metrics.path_count` | 发现的推理路径数 | 7 |
| `metrics.new_nodes_created` | 新创建的命题节点 | 168 |
| `metrics.best_path_judge_confidence` | 最佳路径的评审置信度 | 0.886 |
| `metrics.experiment_exact` | 数值实验精确匹配次数 | 1 |
| `metrics.grade_a_count` | A 级（高质量）推理步骤 | 6 |
| `metrics.OFC` | 总体前沿能力评分 | 64.24 |

### 5.3 belief 的含义

- **belief = 1.0**：命题已被 Lean 形式证明或被设为公理
- **belief > 0.8**：强证据支持，多条推理路径和实验验证
- **belief 0.5–0.8**：有部分支持，但存在未验证的环节
- **belief < 0.5**：推测性命题，证据不足
- **belief = 0.1**：初始目标节点的默认值

---

## 6. 分析真实实验记录（Homochirality Showcase）

仓库中 `evaluate/workspaces/runs/homochirality_showcase/20260328T013808Z/` 保留了一次完整的同手性探索实验。以下是系统的真实发现过程。

### 6.1 三轮推理的思想演进

**第 1 轮（`plausible_prose_attempt_1.txt`）**

系统提出了基于 Turing 不稳定性的空间对称破缺机制。核心思路：利用手性分子不同的扩散系数，在 Frank 模型基础上引发空间不均匀性。

**第 2 轮（`plausible_prose_attempt_2.txt`）**

系统自我纠正了第 1 轮的错误——发现在 Frank 模型的对称结构下，等扩散系数情况不可能产生 Turing 不稳定性（这一点后来被 Lean 形式化验证）。转而探索外部驱动力。

**第 3 轮（`plausible_prose_attempt_3.txt`）**

系统提出了最终机制——**浓度坡道分岔（Concentration Ramp Bifurcation, CRB）**：

1. **初始手性偏差**：利用手性诱导自旋选择效应（CISS）在磁性矿物表面产生 ee₀ ~ 10⁻⁴–10⁻³
2. **动态分岔放大**：溶液浓度因蒸发单调上升，将系统驱过 Frank 模型的分岔点
3. **确定性分支选择**：根据 Berglund-Gentz 动态分岔理论，初始微小偏差在扫越分岔点时被确定性地放大到 |ee| > 0.999

### 6.2 验证链

- **数值实验**（`experiment_code_attempt_1.txt`→`experiment_result_attempt_1.txt`）：
  Python 数值模拟确认了在合理参数下 CRB 机制的可行性

- **Lean 形式化**（`claim_verify_struct_1_code.lean`）：
  证明了等扩散 Frank 系统不具备 Turing 不稳定性

- **定量验证**（`claim_verify_quant_1_code.py`→`claim_verify_quant_1_result.txt`）：
  验证了关键数值声明

### 6.3 最终结果

- 目标 belief：**0.815**（反映了 CISS 表面效率、DKP 互抑制动力学等尚未实验证实的不确定性）
- 发现了 7 条独立推理路径
- 创建了 168 个新命题节点
- 系统在第 2 轮自我否定了第 1 轮的假设（Turing 不稳定性），体现了真正的科学自我纠正能力

---

## 7. 定义你自己的研究问题

### 7.1 创建 Case 文件

```bash
mkdir -p evaluate/cases/your_problem
```

创建 `case.json`：

```json
{
  "case_id": "your_problem",
  "display_name": "你的问题名称",
  "source_proof_config": "proof_config.json",
  "benchmark_scope": "问题的精确描述...",
  "tags": ["math", "open-problem"],
  "model": "cds/Claude-4.6-opus",
  "timeouts": {
    "experiment": 180,
    "decompose": 240,
    "lean": 360
  },
  "lean_policy": {
    "mode": "selective",
    "enable_decomposition": false,
    "enable_strict_lean": false,
    "min_path_confidence": 0.70,
    "max_grade_d_ratio": 0.30
  },
  "planning_constraints": [
    "约束 1：不要假设...",
    "约束 2：必须考虑..."
  ]
}
```

### 7.2 创建 `proof_config.json`

这是最关键的文件——定义种子知识和目标。

**设计原则**：

1. **种子节点**应该是无争议的已知事实（belief=1.0, state="proven"）
2. **目标节点**应该精确描述你希望系统发现/证明的内容
3. **planning_constraints** 应该编码你对问题的领域知识
4. 种子节点数量建议 3–7 个，太少会缺乏上下文，太多会分散注意力

```json
{
  "model": "cds/Claude-4.6-opus",
  "domain": "你的领域",
  "lean_workspace": null,
  "seed_nodes": [
    {
      "key": "fact_1",
      "statement": "已知事实 1 的精确陈述...",
      "belief": 1.0,
      "state": "proven",
      "domain": "具体子领域"
    }
  ],
  "target": {
    "key": "your_target",
    "statement": "目标命题的精确陈述...",
    "belief": 0.1,
    "state": "unverified",
    "domain": "具体子领域"
  },
  "planning_constraints": [
    "领域知识约束..."
  ]
}
```

### 7.3 创建单 Case Suite 并运行

```bash
cat > /tmp/my_suite.json << EOF
{
  "suite_id": "my_experiment",
  "display_name": "我的实验",
  "repeats": 1,
  "run_mode": "serial",
  "cases": ["cases/your_problem/case.json"]
}
EOF

source .env
export PYTHONPATH="$(pwd)/src:$(pwd)"
python scripts/run_benchmark_suite.py \
  --suite /tmp/my_suite.json \
  --output-root evaluate/workspaces
```

### 7.4 Case 设计最佳实践

| 实践 | 说明 |
|------|------|
| 种子节点用教科书级事实 | 避免有争议的假设作为种子 |
| 目标节点要可验证 | "构建一个满足条件 X 的机制"优于"理解 Y" |
| Planning constraints 编码反面知识 | 明确告诉系统哪些路径行不通，避免重复已知死胡同 |
| 设合理的超时 | 实验 120–300s, 分解 180–300s, Lean 240–600s |
| 非数学问题关闭 strict Lean | `enable_strict_lean: false` |

---

## 8. 常见问题与排错

### Q: 报 `ModuleNotFoundError: No module named 'libs.inference_v2'`

PYTHONPATH 未正确设置。确保：

```bash
export PYTHONPATH="$(pwd)/src:$(pwd)"
```

且 `libs/inference_v2/` 目录存在。

### Q: LLM 调用报 401/403

检查 `.env` 中的 `LITELLM_PROXY_API_KEY` 是否正确，以及 `LITELLM_PROXY_API_BASE` 是否可达。

### Q: 运行时间过长

调低 `DISCOVERY_ZERO_MCTS_MAX_TIME_SECONDS`（如设为 3600 = 1 小时）进行快速实验。

### Q: belief 始终很低

可能原因：
- 种子节点太少或与目标关联太弱
- planning_constraints 过于严格
- 问题本身确实超出当前 LLM 的推理能力

### Q: Lean 验证失败

- 确保 `make lean-ready` 已完成
- 检查 `lean_workspace/.lake/` 是否存在
- 非数学问题可设 `enable_strict_lean: false` 跳过

### Q: 如何阅读 `graph.json`

`graph.json` 包含完整的知识超图。关键结构：

```python
from discovery_zero.graph.persistence import load_graph
graph = load_graph("path/to/graph.json")
for node in graph.nodes.values():
    print(f"{node.key}: belief={node.belief:.3f}, state={node.state}")
```

---

## 附录：同手性 Showcase 文件索引

| 文件 | 内容 |
|------|------|
| `PAPER_HOMOCHIRALITY.md` | 完整研究论文（中文） |
| `run_01/summary.json` | 运行指标摘要 |
| `run_01/graph.json` | 168 节点知识超图 |
| `run_01/bridge_plan.json` | 5 步 Bridge 路径 |
| `run_01/exploration_log.json` | 全部 MCTS 迭代日志 |
| `run_01/llm_records/plausible_prose_attempt_*.txt` | 3 轮推测论证（含自我纠正） |
| `run_01/llm_records/experiment_*` | 数值实验代码与结果 |
| `run_01/llm_records/claim_verify_*` | 声明验证记录 |
| `run_01/llm_records/lean_gap_analysis_*.txt` | Lean 差距分析 |
| `run_01/PATH_BENCHMARK_SCORECARD_ZH.md` | 评分卡 |

所有文件均为系统 `20260328T013808Z` 运行的原始输出，未经修改。

> **注意**：`summary.json`、`resolved_proof_config.json` 和 `exploration_log.json` 中记录的路径指向原始运行环境（`/personal/Zero/...`）。这些是历史元数据，保留原样以确保实验记录的完整性，不影响新的运行。
