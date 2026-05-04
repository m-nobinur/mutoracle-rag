.PHONY: install format lint type test check cli

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
