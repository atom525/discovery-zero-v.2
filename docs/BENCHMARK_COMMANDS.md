# Benchmark 运行命令参考

## 环境配置

项目根目录下创建 `.env` 文件（从 `.env.example` 复制）：

```bash
cp .env.example .env
# 编辑 .env 填入真实 API 密钥
```

设置 Python 路径（使用仓库内置依赖）：

```bash
export PYTHONPATH="$(pwd)/src:$(pwd)"
```

## 运行基准套件

```bash
source .env
export PYTHONPATH="$(pwd)/src:$(pwd)"

python scripts/run_benchmark_suite.py \
  --suite evaluate/suite.json \
  --repeats 1 \
  --output-root evaluate/workspaces
```

## 运行结果

运行输出位于：
- `evaluate/workspaces/runs/<suite_name>/<timestamp>/`
- `evaluate/workspaces/reports/<suite_name>/`

每个 case 目录包含：
- `graph.json` -- 最终知识图谱快照
- `exploration_log.json` -- MCTS 步骤记录
- `bridge_plan.json` -- Bridge 规划
- `summary.json` -- 指标摘要
- `llm_records/` -- LLM 输出记录
