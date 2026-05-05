.PHONY: install format lint type test check cli smoke mutate data baseline

install:
	uv sync --all-extras --dev

format:
	uv run ruff format .

lint:
	uv run ruff check .

type:
	uv run mypy src/mutoracle

test:
	uv run pytest

check: lint type test

cli:
	uv run mutoracle --help

smoke:
	uv run mutoracle smoke --queries 10

mutate:
	uv run mutoracle mutate --operator CI

data:
	uv run mutoracle data build

baseline:
	uv run python experiments/run_baselines.py --baseline metarag --queries 2
