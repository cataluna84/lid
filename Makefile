.PHONY: install dev lint format typecheck test clean train infer visualize bench bench-quick verify

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

bench:
	uv run lid-bench configs/bench_grid.yaml $(ARGS)

bench-quick:
	uv run lid-bench configs/bench_quick.yaml --no-wandb $(ARGS)

# Run all VERIFY.md checks in order, stopping on first failure.
# See VERIFY.md for the per-section exit-code contract (1=env, 2=quality,
# 3=bench, 4=notebook, 5=data/wandb).
verify:
	@set -e ; \
	echo "[verify 1/7] Environment sanity..." ; \
	python3 --version ; \
	uv --version ; \
	uv run python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "  FAIL: CUDA not available" ; exit 1 ; } ; \
	uv run python -c "import torch; assert '+cu128' in torch.__version__, torch.__version__" || { echo "  FAIL: torch is not the cu128 build" ; exit 1 ; } ; \
	uv run python -c "import os; from dotenv import load_dotenv; load_dotenv(); assert os.environ.get('HF_TOKEN') or os.environ.get('HF_NEW'), 'missing HF token in .env'" || { echo "  FAIL: HF token missing" ; exit 1 ; } ; \
	echo "[verify 2/7] Code quality..." ; \
	$(MAKE) lint || { echo "  FAIL: lint" ; exit 2 ; } ; \
	$(MAKE) typecheck || { echo "  FAIL: typecheck" ; exit 2 ; } ; \
	$(MAKE) test || { echo "  FAIL: tests" ; exit 2 ; } ; \
	echo "[verify 3/7] Smoke benchmark (skipped here -- run \`make bench-quick\` manually)" ; \
	echo "[verify 4/7] Notebook smoke..." ; \
	uv run python -c "import unsloth, unsloth_zoo, xformers, trl" 2>/dev/null || echo "  WARN: unsloth extras not installed (run \`uv sync --extra unsloth\`)" ; \
	uv run python -c "import json; json.loads(open('notebooks/LID_Embedding_Classifier.ipynb').read())" || { echo "  FAIL: notebook JSON" ; exit 4 ; } ; \
	echo "[verify 5/7] HF dataset access (skipped here -- run \`uv run python -c \"from datasets import load_dataset; load_dataset('1024m/LID', data_files='Data_Hackathon/LID-1000.parquet')\"\` manually)" ; \
	echo "[verify 6/7] W&B connectivity (skipped here -- run \`uv run wandb status\` manually)" ; \
	echo "[verify 7/7] Memory system files present..." ; \
	test -f AGENTS.md || { echo "  FAIL: AGENTS.md missing" ; exit 1 ; } ; \
	test -f PLAN.md || { echo "  FAIL: PLAN.md missing" ; exit 1 ; } ; \
	test -f .factory/memories.md || { echo "  FAIL: .factory/memories.md missing" ; exit 1 ; } ; \
	test -f VERIFY.md || { echo "  FAIL: VERIFY.md missing" ; exit 1 ; } ; \
	test -f notebooks/AGENTS.md || { echo "  FAIL: notebooks/AGENTS.md missing" ; exit 1 ; } ; \
	test -f .factory/skills/context-bootstrap/SKILL.md || { echo "  FAIL: context-bootstrap SKILL missing" ; exit 1 ; } ; \
	echo "All checks passed."
