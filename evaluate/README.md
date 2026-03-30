# Evaluate 目录说明

本目录用于运行 Discovery-Zero-v2 的基准评测。

## 保留内容

- `suite.json`：当前唯一保留的完整评测套件
- `cases/`：各问题的 case 与 proof_config

## 运行方式

```bash
cd /personal/Discovery-Zero-v2
python scripts/run_benchmark_suite.py \
  --suite evaluate/suite.json \
  --repeats 1 \
  --output-root evaluate/workspaces
```

## 输出目录

运行输出会写入：

- `evaluate/workspaces/runs/`
- `evaluate/workspaces/reports/`

这两个目录已在 `.gitignore` 中忽略，不应提交到仓库。
