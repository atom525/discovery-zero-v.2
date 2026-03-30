# Discovery Zero - Makefile
# 提供 Lean 环境配置等常用命令

.PHONY: install test lean-ready lean-verify clean help

help:
	@echo "Discovery Zero - 可用命令:"
	@echo "  make install       - 安装 Python 包"
	@echo "  make test          - 运行测试"
	@echo "  make lean-ready    - Lean 一键配置（自包含）"
	@echo "  make lean-verify   - 验证 lean_workspace (lake build)"
	@echo "  make clean         - 清理构建产物"
	@echo ""
	@echo "详见 README.md"

install:
	pip install -e ".[dev]"

test:
	pytest -v --tb=short

# 一键完成（自包含）
lean-ready:
	chmod +x scripts/setup_lean_full.sh
	./scripts/setup_lean_full.sh

# 验证 lean_workspace
lean-verify:
	cd lean_workspace && lake build

clean:
	rm -rf .pytest_cache build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf lean_workspace/.lake lean_workspace/**/*.olean 2>/dev/null || true
