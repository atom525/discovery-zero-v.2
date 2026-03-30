# Discovery Zero 基准测试运行指南

## 环境准备

### 1. 配置 `.env`

项目根目录 `/personal/Zero/.env`：

```bash
LITELLM_PROXY_API_BASE=https://ai-gateway-internal.dp.tech
LITELLM_PROXY_API_KEY=sk-A7BufOlwJzf2dh81I7YMMw
DISCOVERY_ZERO_ENABLE_MCTS=1
```

### 2. 环境变量

```bash
export PYTHONPATH="/personal/Zero/src:/personal/Gaia"
```

## 运行命令

### 标准双 case 基准测试（screen 后台）

```bash
screen -S benchmark -dm bash -c '
set -a
source /personal/Zero/.env
set +a
export PYTHONPATH="/personal/Zero/src:/personal/Gaia"
cd /personal/Zero
exec python -u -c "
import sys
sys.argv = [\"dz\", \"benchmark\", \"run-suite\", \"--suite\", \"evaluate/suite_two_cases.json\", \"--output-root\", \"evaluate/workspaces\"]
from discovery_zero.cli import app
app()
"
'
```

### 带自定义参数运行

通过环境变量覆盖默认配置：

```bash
screen -S benchmark -dm bash -c '
set -a
source /personal/Zero/.env
set +a
export PYTHONPATH="/personal/Zero/src:/personal/Gaia"
export DISCOVERY_ZERO_MCTS_MAX_TIME_SECONDS=7200    # 每个 case 最大 2 小时
export DISCOVERY_ZERO_MCTS_MAX_ITERATIONS=50         # 最大 50 次 MCTS 迭代
cd /personal/Zero
exec python -u -c "
import sys
sys.argv = [\"dz\", \"benchmark\", \"run-suite\", \"--suite\", \"evaluate/suite_two_cases.json\", \"--output-root\", \"evaluate/workspaces\"]
from discovery_zero.cli import app
app()
"
'
```

## 监控

```bash
# 查看 screen 列表
screen -ls

# 进入 screen 查看实时输出
screen -r benchmark

# 退出 screen（不中断运行）：Ctrl+A, D

# 查看最新 run 目录
ls -lt evaluate/workspaces/runs/two_cases_mcts/ | head -5

# 查看 LLM 记录文件
ls -lt evaluate/workspaces/runs/two_cases_mcts/<TIMESTAMP>/<case>/run_01/llm_records/

# 查看 exploration log 的迭代进度
grep '"last_iteration"' evaluate/workspaces/runs/two_cases_mcts/<TIMESTAMP>/<case>/run_01/exploration_log.json

# 查看 action_module 分布
grep '"action_module"' evaluate/workspaces/runs/two_cases_mcts/<TIMESTAMP>/<case>/run_01/exploration_log.json
```

## 查看结果

```bash
# Scorecard（中文摘要）
cat evaluate/workspaces/runs/two_cases_mcts/<TIMESTAMP>/<case>/run_01/PATH_BENCHMARK_SCORECARD_ZH.md

# 详细指标
cat evaluate/workspaces/runs/two_cases_mcts/<TIMESTAMP>/<case>/run_01/summary.json

# Bridge plan
cat evaluate/workspaces/runs/two_cases_mcts/<TIMESTAMP>/<case>/run_01/bridge_plan.json

# Graph
cat evaluate/workspaces/runs/two_cases_mcts/<TIMESTAMP>/<case>/run_01/graph.json
```

## 单元测试

```bash
cd /personal/Zero

# 全量测试
pytest -q tests/

# 仅 MCTS 相关测试
pytest -q tests/planning/test_mcts_route_feedback_and_injection.py
pytest -q tests/planning/test_mcts_verification_reward.py
pytest -q tests/planning/test_mcts_integration.py

# Bridge 连通性测试
pytest -q tests/planning/test_bridge_connectivity.py

# 跳过需要 LLM/Lean 的测试
pytest -q -m "not llm and not lean" tests/
```

## 关键配置参数

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `DISCOVERY_ZERO_ENABLE_MCTS` | `false` | 启用 MCTS 引擎（必须设为 1） |
| `DISCOVERY_ZERO_MCTS_MAX_ITERATIONS` | `50` | 最大 MCTS 迭代次数 |
| `DISCOVERY_ZERO_MCTS_MAX_TIME_SECONDS` | `3600`（.env 中设为 `21600`） | 每个 case 最大秒数（6 小时） |
| `DISCOVERY_ZERO_MCTS_C_PUCT` | `1.4` | UCB 探索系数 |
| `DISCOVERY_ZERO_ENABLE_ANALOGY` | `true` | 启用类比引擎 |
| `DISCOVERY_ZERO_ENABLE_SPECIALIZE` | `true` | 启用特化引擎 |
| `DISCOVERY_ZERO_ENABLE_DECOMPOSE` | `true` | 启用分解引擎 |
| `DISCOVERY_ZERO_ENABLE_CLAIM_VERIFIER` | `true` | 启用 claim 验证 |
| `DISCOVERY_ZERO_VERIFICATION_PARALLEL_WORKERS` | `3` | claim 并行验证线程数 |
| `DISCOVERY_ZERO_BP_PROPAGATION_THRESHOLD` | `3` | BP 传播触发阈值 |

## Suite 配置

Suite 文件路径：`evaluate/suite_two_cases.json`

包含的 case：
- `lonely_runner_n11`：孤独跑者猜想 n=11
- `furstenberg_times2_times3`：Furstenberg ×2×3 猜想（零熵情形）

Case 配置位于 `evaluate/cases/<case_id>/case_frontier_assisted.json`，可自定义模型、lean_policy、timeouts 等。
