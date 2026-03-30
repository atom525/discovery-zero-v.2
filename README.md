# Discovery-Zero-v2

Discovery Zero 是一个面向开放问题探索的数学发现系统。系统用“推理超图”统一表示命题、推理步骤、验证结果，并通过信念传播驱动下一步探索。

## 1. 项目结构

- `src/discovery_zero/`：核心源码
  - `graph/`：图模型、注入、推理适配
  - `planning/`：MCTS/编排/桥接规划
  - `tools/`：LLM、Lean、实验等工具封装
- `src/gaia_bp/`：内置的 Gaia 信念传播引擎（从 Gaia 仓库 vendor 并扩展）
- `libs/`：推理兼容层（`inference_v2/` shim、`graph_ir/`、`storage/`、`embedding`）
- `tests/`：测试用例
- `evaluate/`：基准 case 与 suite 配置
- `lean_workspace/`：Lean 工程（源码保留，构建产物已移除）
- `scripts/`：Lean 安装与环境脚本
- `docs/`：设计与架构文档

## 2. 环境要求

- Python `>= 3.12`
- 仓库已内置 Gaia 依赖组件（无需额外 Gaia 仓库）
- 若使用 LLM 功能，需要可用的 LiteLLM 兼容网关

## 3. 安装

```bash
cd /personal/Discovery-Zero-v2
pip install -e ".[dev]"
```

运行前请设置：

```bash
export PYTHONPATH="$(pwd)/src:$(pwd):$PYTHONPATH"
```

## 4. 快速使用

### 4.1 查看 CLI

```bash
dz --help
```

### 4.2 初始化一个超图

```bash
dz init --path graph.json
```

### 4.3 运行测试

```bash
pytest -v --tb=short
# 或
make test
```

### 4.4 运行基准

```bash
python scripts/run_benchmark_suite.py --suite evaluate/suite.json
```

## 5. Lean 环境

推荐命令：

```bash
make lean-ready
```

常用命令：

```bash
make lean-ready     # 一键安装并初始化 Lean（推荐）
make lean-verify    # 在 lean_workspace 执行 lake build
```

## 6. LLM 配置

请复制并填写环境变量：

```bash
cp .env.example .env
```

最小配置项：

- `LITELLM_PROXY_API_BASE`
- `LITELLM_PROXY_API_KEY`
- `DISCOVERY_ZERO_LLM_MODEL`（可选）

注意：仓库中不应提交真实密钥。

## 7. 协作开发约定

- 功能代码放在 `src/discovery_zero/`，避免把实验产物写入仓库。
- 新增能力需补充对应测试到 `tests/`。
- 中间运行数据统一放本地工作目录，不提交 `evaluate/workspaces/runs/`。
- 提交前建议执行：

```bash
pytest -q
```

## 8. 研究流程指南

完整的从零跑通指南见 **[docs/WALKTHROUGH_ZH.md](docs/WALKTHROUGH_ZH.md)**，以同手性机制（Homochirality）这一真实案例为范例，覆盖：

- 环境配置 → Case 设计 → 运行探索 → 解读产物
- 真实实验记录分析（`evaluate/workspaces/runs/homochirality_showcase/`）
- 自定义研究问题模板

## 9. Showcase：同手性机制发现

仓库包含一份完整的真实实验记录（`evaluate/workspaces/runs/homochirality_showcase/20260328T013808Z/`），展示 Discovery Zero 对"前生物同手性起源"这一开放问题的自主探索过程。系统发现了 **浓度坡道分岔（CRB）机制**，最终 belief 达到 0.815。

详见 `PAPER_HOMOCHIRALITY.md` 和 `run_01/` 目录下的完整 LLM 交互记录。

## 10. 当前仓库清理说明

本仓库已移除以下内容以便协作与推送：

- 中间运行数据（`evaluate/workspaces/runs/`，showcase 除外）
- 生成报告（`evaluate/workspaces/reports/`）
- 缓存与构建产物（`__pycache__/`, `.pytest_cache/`, `lean_workspace/.lake/`）
- 敏感配置（`.env`）
- 内部 AI 辅助上下文文件

