.PHONY: install format lint type test check cli smoke mutate data baseline experiment-smoke experiment-dev experiment-full localization-calibrated analysis analysis-dev analysis-smoke release-check release-check-full paper-pdf

UV_CACHE_DIR ?= .uv-cache
UV := UV_CACHE_DIR=$(UV_CACHE_DIR) uv

install:
	$(UV) sync --all-extras --dev

format:
	$(UV) run ruff format .

lint:
	$(UV) run ruff check .

type:
	$(UV) run mypy src/mutoracle

test:
	$(UV) run pytest

check: lint type test

cli:
	$(UV) run mutoracle --help

smoke:
	$(UV) run mutoracle smoke --queries 10

mutate:
	$(UV) run mutoracle mutate --operator CI

data:
	$(UV) run mutoracle data build

baseline:
	$(UV) run python experiments/run_baselines.py --baseline metarag --queries 2

experiment-smoke:
	$(UV) run python experiments/run_baselines.py --experiment-config experiments/configs/e1_detection.yaml --smoke
	$(UV) run python experiments/run_mutoracle.py --config experiments/configs/e2_localization.yaml --smoke
	$(UV) run python experiments/run_ablation.py --config experiments/configs/e3_ablation.yaml --smoke
	$(UV) run python experiments/run_ablation.py --config experiments/configs/e4_separability.yaml --smoke
	$(UV) run python experiments/run_latency.py --config experiments/configs/e5_latency.yaml --smoke
	$(UV) run python experiments/run_ablation.py --config experiments/configs/e6_weighted.yaml --smoke

experiment-dev:
	$(UV) run python experiments/run_baselines.py --experiment-config experiments/configs/e1_detection.yaml --mode dev --progress-every 5
	$(UV) run python experiments/run_mutoracle.py --config experiments/configs/e2_localization.yaml --mode dev --progress-every 5
	$(UV) run python experiments/run_ablation.py --config experiments/configs/e3_ablation.yaml --mode dev --progress-every 5
	$(UV) run python experiments/run_ablation.py --config experiments/configs/e4_separability.yaml --mode dev --progress-every 5
	$(UV) run python experiments/run_latency.py --config experiments/configs/e5_latency.yaml --mode dev --progress-every 5
	$(UV) run python experiments/run_ablation.py --config experiments/configs/e6_weighted.yaml --mode dev --progress-every 5

experiment-full:
	$(UV) run python experiments/run_baselines.py --experiment-config experiments/configs/e1_detection.yaml --mode full
	$(UV) run python experiments/run_mutoracle.py --config experiments/configs/e2_localization.yaml --mode full
	$(UV) run python experiments/run_ablation.py --config experiments/configs/e3_ablation.yaml --mode full
	$(UV) run python experiments/run_ablation.py --config experiments/configs/e4_separability.yaml --mode full
	$(UV) run python experiments/run_latency.py --config experiments/configs/e5_latency.yaml --mode full
	$(UV) run python experiments/run_ablation.py --config experiments/configs/e6_weighted.yaml --mode full

localization-calibrated:
	$(UV) run python experiments/run_calibrated_localization.py

analysis:
	$(UV) run python experiments/run_calibrated_localization.py
	$(UV) run python experiments/analyze_results.py --mode full --paper-dir .local/analysis-assets --duckdb-path experiments/results/analysis.duckdb

analysis-dev:
	$(UV) run python experiments/analyze_results.py --mode dev --paper-dir .local/analysis-assets

analysis-smoke:
	$(UV) run python experiments/analyze_results.py --mode smoke --paper-dir .local/analysis-assets --duckdb-path experiments/results/analysis-smoke.duckdb

release-check:
	$(UV) run mutoracle release-check

release-check-full:
	$(UV) run mutoracle release-check --strict-full-results

paper-pdf:
	@command -v latexmk >/dev/null 2>&1 || { echo "latexmk is not installed. Install MacTeX or TinyTeX to build paper/main.tex locally."; exit 2; }
	@command -v inkscape >/dev/null 2>&1 || { echo "inkscape is not installed. Install Inkscape to enable SVG figure conversion for paper/main.tex."; exit 2; }
	$(UV) run latexmk -g -cd -pdf -shell-escape -interaction=nonstopmode paper/main.tex
