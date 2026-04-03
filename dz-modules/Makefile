PYTHON ?= python
PIP ?= $(PYTHON) -m pip

.PHONY: install-all test lint

install-all:
	$(PIP) install -e /personal/gaia-bp
	$(PIP) install -e packages/dz-hypergraph
	$(PIP) install -e packages/dz-verify
	$(PIP) install -e packages/dz-engine
	$(PIP) install -e packages/dz-mcp
	$(PIP) install pytest pytest-cov ruff

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check packages tests
