.PHONY: install dev lint format typecheck test clean train infer visualize

install:
	uv sync

dev:
	uv sync --group dev
	uv run pre-commit install

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

typecheck:
	uv run mypy src/lid/

test:
	uv run pytest

clean:
	rm -rf outputs/ checkpoints/ lang_plots/ .mypy_cache/ .pytest_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

train:
	uv run lid-train $(ARGS)

infer:
	uv run lid-infer $(ARGS)

visualize:
	uv run lid-visualize $(ARGS)
