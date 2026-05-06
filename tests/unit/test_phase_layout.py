from __future__ import annotations

from pathlib import Path

from mutoracle.aggregation import (
    ConfidenceGatedAggregator,
    UniformAggregator,
    WeightedAggregator,
)
from mutoracle.baselines import MetaRAGBaseline, RagasBaseline, run_baselines
from mutoracle.data import build_fits_dataset, validate_fits_records
from mutoracle.localizer import FaultLocalizer
from mutoracle.mutations import get_operator, mutation_registry
from mutoracle.oracles import LLMJudgeOracle, NLIOracle, SemanticSimilarityOracle
from mutoracle.pipeline.prompt import build_rag_prompt
from mutoracle.pipeline.rag import FixtureRAGPipeline
from mutoracle.pipeline.retriever import LexicalRetriever
from mutoracle.providers.openrouter_provider import OpenRouterProvider
from mutoracle.storage.sqlite_cache import SQLiteCacheLedger


def test_planned_experiment_entry_points_exist() -> None:
    assert Path("experiments/run_baselines.py").exists()
    assert Path("experiments/run_weight_search.py").exists()
    assert Path("experiments/run_mutoracle.py").exists()
    assert Path("experiments/run_ablation.py").exists()
    assert Path("experiments/run_latency.py").exists()
    for name in (
        "e1_detection.yaml",
        "e2_localization.yaml",
        "e3_ablation.yaml",
        "e4_separability.yaml",
        "e5_latency.yaml",
        "e6_weighted.yaml",
    ):
        assert Path("experiments/configs", name).exists()
    assert Path("docs/EXPERIMENTS.md").exists()


def test_phase_ten_release_artifacts_exist() -> None:
    for path in (
        "LICENSE",
        "README.md",
        "Makefile",
        "pyproject.toml",
        "data/manifests/datasets.json",
        "data/fits/manifest.json",
        "experiments/configs/e1_detection.yaml",
        "experiments/configs/e2_localization.yaml",
        "experiments/configs/e3_ablation.yaml",
        "experiments/configs/e4_separability.yaml",
        "experiments/configs/e5_latency.yaml",
        "experiments/configs/e6_weighted.yaml",
    ):
        assert Path(path).exists()


def test_phase_two_module_layout_exports_expected_symbols() -> None:
    assert build_rag_prompt
    assert FixtureRAGPipeline
    assert LexicalRetriever
    assert OpenRouterProvider
    assert SQLiteCacheLedger


def test_phase_three_module_layout_exports_expected_symbols() -> None:
    assert get_operator("CI")
    assert set(mutation_registry()) == {
        "CI",
        "CR",
        "CS",
        "QP",
        "QN",
        "QD",
        "QI",
        "FS",
        "FA",
        "FE",
        "GN",
    }


def test_phase_four_module_layout_exports_expected_symbols() -> None:
    assert SemanticSimilarityOracle
    assert NLIOracle
    assert LLMJudgeOracle


def test_phase_five_module_layout_exports_expected_symbols() -> None:
    assert UniformAggregator
    assert WeightedAggregator
    assert ConfidenceGatedAggregator
    assert FaultLocalizer


def test_phase_six_module_layout_exports_expected_symbols() -> None:
    assert build_fits_dataset
    assert validate_fits_records


def test_phase_seven_module_layout_exports_expected_symbols() -> None:
    assert RagasBaseline
    assert MetaRAGBaseline
    assert run_baselines
