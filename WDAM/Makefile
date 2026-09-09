.PHONY: lint format compile check clean help

PYTHON ?= python

help:
	@echo "make lint     - check code with ruff"
	@echo "make format   - auto-format code with ruff"
	@echo "make compile  - syntax-check Python sources with compileall"
	@echo "make check    - compile + lint"
	@echo "make clean    - remove Python cache files"

lint:
	$(PYTHON) -m ruff check openwam/ scripts/ benchmarks/

format:
	$(PYTHON) -m ruff format openwam/ scripts/ benchmarks/
	$(PYTHON) -m ruff check --fix openwam/ scripts/ benchmarks/

compile:
	$(PYTHON) -m compileall -q openwam scripts benchmarks

check: compile lint

clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	find . -name "*.pyc" -delete
