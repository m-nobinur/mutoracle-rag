.PHONY: install format lint type test check cli smoke mutate data baseline experiment-smoke experiment-dev experiment-full analysis analysis-dev

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

experiment-smoke:
	uv run python experiments/run_baselines.py --experiment-config experiments/configs/e1_detection.yaml --smoke
	uv run python experiments/run_mutoracle.py --config experiments/configs/e2_localization.yaml --smoke
	uv run python experiments/run_ablation.py --config experiments/configs/e3_ablation.yaml --smoke
	uv run python experiments/run_ablation.py --config experiments/configs/e4_separability.yaml --smoke
	uv run python experiments/run_latency.py --config experiments/configs/e5_latency.yaml --smoke
	uv run python experiments/run_ablation.py --config experiments/configs/e6_weighted.yaml --smoke

experiment-dev:
	uv run python experiments/run_baselines.py --experiment-config experiments/configs/e1_detection.yaml --mode dev --progress-every 5
	uv run python experiments/run_mutoracle.py --config experiments/configs/e2_localization.yaml --mode dev --progress-every 5
	uv run python experiments/run_ablation.py --config experiments/configs/e3_ablation.yaml --mode dev --progress-every 5
	uv run python experiments/run_ablation.py --config experiments/configs/e4_separability.yaml --mode dev --progress-every 5
	uv run python experiments/run_latency.py --config experiments/configs/e5_latency.yaml --mode dev --progress-every 5
	uv run python experiments/run_ablation.py --config experiments/configs/e6_weighted.yaml --mode dev --progress-every 5

experiment-full:
	uv run python experiments/run_baselines.py --experiment-config experiments/configs/e1_detection.yaml --mode full
	uv run python experiments/run_mutoracle.py --config experiments/configs/e2_localization.yaml --mode full
	uv run python experiments/run_ablation.py --config experiments/configs/e3_ablation.yaml --mode full
	uv run python experiments/run_ablation.py --config experiments/configs/e4_separability.yaml --mode full
	uv run python experiments/run_latency.py --config experiments/configs/e5_latency.yaml --mode full
	uv run python experiments/run_ablation.py --config experiments/configs/e6_weighted.yaml --mode full

analysis:
	uv run python experiments/analyze_results.py

analysis-dev:
	uv run python experiments/analyze_results.py --mode dev
