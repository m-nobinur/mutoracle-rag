from __future__ import annotations

import json
import sys
from importlib import util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from mutoracle.cache import UsageSummary
from mutoracle.config import MutOracleConfig
from mutoracle.contracts import RAGRun
from mutoracle.experiments import (
    ExperimentRunSettings,
    artifact_paths,
    build_experiment_aggregator,
    enforce_cost_gate,
    ensure_full_run_allowed,
    estimate_cost_usd,
    expected_detection_label,
    expected_diagnosis_stage,
    fixture_model_ids,
    load_experiment_config,
    load_runtime_config,
    print_cost_estimate,
    provider_route_for_oracles,
    rag_run_from_fits_record,
    real_model_ids,
    real_oracles,
    resolve_oracle_mode,
    resolve_run_settings,
    resolve_runtime_config_path,
    selected_fits_records,
    usage_delta,
)


def test_phase_eight_config_suite_exists() -> None:
    for name in (
        "e1_detection.yaml",
        "e2_localization.yaml",
        "e3_ablation.yaml",
        "e4_separability.yaml",
        "e5_latency.yaml",
        "e6_weighted.yaml",
    ):
        assert Path("experiments/configs", name).exists()


def test_cost_gate_blocks_over_cap_without_confirmation(tmp_path: Path) -> None:
    settings = ExperimentRunSettings(
        experiment_id="cost_gate_test",
        title="Cost gate test",
        mode="smoke",
        config_path=tmp_path / "config.yaml",
        dataset_path=Path("data/fits/fits_v1.0.0/fits.jsonl"),
        split="test",
        query_limit=2,
        seeds=[13, 42, 91],
        output_dir=tmp_path,
        cost_cap_usd=0.01,
        estimated_cost_per_example_usd=0.01,
        require_smoke_before_full=True,
    )

    estimated = estimate_cost_usd(settings, work_units_per_record=1)
    with pytest.raises(RuntimeError, match="Cost gate blocked"):
        enforce_cost_gate(
            settings,
            estimated_cost_usd=estimated,
            confirm_cost=False,
        )

    enforce_cost_gate(settings, estimated_cost_usd=estimated, confirm_cost=True)


def test_cost_gate_enforces_global_five_usd_confirmation(
    tmp_path: Path,
) -> None:
    settings = ExperimentRunSettings(
        experiment_id="policy_cap_test",
        title="Policy cap test",
        mode="smoke",
        config_path=tmp_path / "config.yaml",
        dataset_path=Path("data/fits/fits_v1.0.0/fits.jsonl"),
        split="test",
        query_limit=1,
        seeds=[13],
        output_dir=tmp_path,
        cost_cap_usd=10.0,
        estimated_cost_per_example_usd=6.0,
        require_smoke_before_full=True,
    )

    estimated = estimate_cost_usd(settings, work_units_per_record=1)
    with pytest.raises(RuntimeError, match="Cost gate blocked"):
        enforce_cost_gate(
            settings,
            estimated_cost_usd=estimated,
            confirm_cost=False,
        )


def test_mutoracle_smoke_writes_phase_eight_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = tmp_path / "fits.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "qid": "rgbp8_test_0001",
                "query": "Who wrote the notes?",
                "gt_answer": "Ada Lovelace",
                "fault_stage": "generation",
                "split": "test",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "e2_smoke.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  id: e2_test",
                "  title: E2 test",
                f"  output_dir: {tmp_path.as_posix()}",
                "dataset:",
                f"  path: {dataset_path.as_posix()}",
                "  split: test",
                "smoke:",
                "  query_limit: 1",
                "  seeds: [13, 42, 91]",
                "cost_gate:",
                "  estimated_cost_per_example_usd: 0.0",
                "  max_estimated_cost_usd: 0.5",
                "localizer:",
                "  aggregation: weighted",
                "  weights:",
                "    nli: 0.4",
                "    semantic_similarity: 0.3",
                "    llm_judge: 0.3",
                "  delta_threshold: 0.05",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_mutoracle.py",
            "--config",
            str(config_path),
            "--mode",
            "smoke",
        ],
    )

    _load_script("experiments/run_mutoracle.py").main()

    settings = ExperimentRunSettings(
        experiment_id="e2_test",
        title="E2 test",
        mode="smoke",
        config_path=config_path,
        dataset_path=dataset_path,
        split="test",
        query_limit=1,
        seeds=[13, 42, 91],
        output_dir=tmp_path,
        cost_cap_usd=0.5,
        estimated_cost_per_example_usd=0.0,
        require_smoke_before_full=True,
    )
    paths = artifact_paths(settings)
    assert paths.raw_jsonl.exists()
    assert paths.summary_csv.exists()
    assert paths.config_snapshot_yaml.exists()
    assert paths.failures_jsonl.exists()
    assert paths.duckdb_sql.exists()

    manifest = json.loads(paths.manifest_json.read_text(encoding="utf-8"))
    assert manifest["seeds"] == [13, 42, 91]
    assert manifest["row_count"] == 3
    assert manifest["config_snapshot_yaml"] == str(paths.config_snapshot_yaml)
    assert manifest["run_id"]
    assert manifest["git_commit"]
    assert manifest["dataset_checksum"]
    assert manifest["duckdb_sql"] == str(paths.duckdb_sql)
    assert "python" in manifest["sdk_versions"]
    assert "fixture-fits-generator" in manifest["model_ids"]
    assert set(manifest["provider_routing"]) == {"fixture"}


def test_expected_stage_and_label_support_explicit_fields() -> None:
    record = {
        "expected_stage": "no_fault_detected",
        "expected_label": "faithful",
        "fault_stage": "generation",
    }

    assert expected_diagnosis_stage(record) == "no_fault_detected"
    assert expected_detection_label(record) == "faithful"


def test_record_selection_balances_ordered_stage_splits(tmp_path: Path) -> None:
    dataset_path = tmp_path / "ordered.jsonl"
    rows = []
    for stage in ("retrieval", "prompt", "generation", "no_fault"):
        for index in range(4):
            rows.append(
                {
                    "qid": f"{stage}_{index}",
                    "query": "q",
                    "gt_answer": "a",
                    "fault_stage": stage,
                    "split": "test",
                }
            )
    dataset_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    settings = ExperimentRunSettings(
        experiment_id="balanced_selection",
        title="Balanced selection",
        mode="dev",
        config_path=tmp_path / "config.yaml",
        dataset_path=dataset_path,
        split="test",
        query_limit=8,
        seeds=[13],
        output_dir=tmp_path,
        cost_cap_usd=0.5,
        estimated_cost_per_example_usd=0.0,
        require_smoke_before_full=True,
    )

    selected = selected_fits_records(settings)

    assert [row["fault_stage"] for row in selected] == [
        "retrieval",
        "prompt",
        "generation",
        "no_fault",
        "retrieval",
        "prompt",
        "generation",
        "no_fault",
    ]


def test_rag_run_materialization_uses_staged_passages() -> None:
    record = {
        "qid": "rgbp8_000000",
        "query": "When is the premiere?",
        "gt_answer": "January 2, 2022",
        "fault_stage": "retrieval",
        "split": "validation",
        "source": "rgb",
        "source_qid": "rgb_000000",
        "supporting_passage": "A supporting passage.",
        "distractor_passage": "A distractor passage.",
        "generation_model": "fixture-rgb-generator",
        "provider_route": "fixture",
    }

    run = rag_run_from_fits_record(record, seed=13)
    assert run.passages == ["A distractor passage."]
    assert run.metadata["generation"]["model"] == "fixture-rgb-generator"
    assert run.metadata["generation"]["provider_route"] == "fixture"


def test_usage_delta_is_non_negative() -> None:
    before = UsageSummary(
        requests=10,
        live_requests=4,
        cache_hits=6,
        prompt_tokens=100,
        completion_tokens=80,
        total_cost_usd=0.2,
        total_latency_seconds=1.5,
    )
    after = UsageSummary(
        requests=13,
        live_requests=6,
        cache_hits=7,
        prompt_tokens=160,
        completion_tokens=110,
        total_cost_usd=0.32,
        total_latency_seconds=2.2,
    )

    delta = usage_delta(before, after)
    assert delta["requests"] == 3
    assert delta["prompt_tokens"] == 60
    assert delta["completion_tokens"] == 30
    assert delta["total_tokens"] == 90
    assert float(delta["cost_usd"]) == pytest.approx(0.12)


def test_experiment_config_validation_branches(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("- not a mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_experiment_config(invalid)

    bad_limit = tmp_path / "bad-limit.yaml"
    bad_limit.write_text(
        "\n".join(
            [
                "experiment: {id: bad}",
                "smoke:",
                "  query_limit: 0",
                "  seeds: [13]",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="query_limit"):
        resolve_run_settings(bad_limit, mode="smoke", default_experiment_id="bad")

    no_seeds = tmp_path / "no-seeds.yaml"
    no_seeds.write_text(
        "\n".join(
            [
                "experiment: {id: bad}",
                "smoke:",
                "  query_limit: 1",
                "  seeds: []",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="At least one seed"):
        resolve_run_settings(no_seeds, mode="smoke", default_experiment_id="bad")

    bad_section = tmp_path / "bad-section.yaml"
    bad_section.write_text("experiment: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected config section"):
        resolve_run_settings(bad_section, mode="smoke", default_experiment_id="bad")


def test_selected_records_and_label_mapping_branches(tmp_path: Path) -> None:
    dataset = tmp_path / "records.jsonl"
    dataset.write_text(
        "\n".join(
            [
                "",
                json.dumps(
                    {"qid": "1", "split": "validation", "fault_stage": "no_fault"}
                ),
                json.dumps({"qid": "2", "split": "test", "fault_stage": "prompt"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    settings = ExperimentRunSettings(
        experiment_id="select",
        title="select",
        mode="smoke",
        config_path=tmp_path / "config.yaml",
        dataset_path=dataset,
        split="all",
        query_limit=2,
        seeds=[13],
        output_dir=tmp_path,
        cost_cap_usd=1.0,
        estimated_cost_per_example_usd=0.0,
        require_smoke_before_full=True,
    )

    assert [row["qid"] for row in selected_fits_records(settings)] == ["1", "2"]
    empty = settings.__class__(
        **{**settings.__dict__, "split": "missing", "query_limit": 1}
    )
    with pytest.raises(ValueError, match="No experiment records"):
        selected_fits_records(empty)

    assert expected_diagnosis_stage({"fault_stage": "no_fault"}) == "no_fault_detected"
    assert (
        expected_diagnosis_stage({"fault_stage": "no_fault_detected"})
        == "no_fault_detected"
    )
    assert expected_diagnosis_stage({"fault_stage": "retrieval"}) == "retrieval"
    with pytest.raises(ValueError, match="Unsupported expected_stage"):
        expected_diagnosis_stage({"expected_stage": "bad"})
    with pytest.raises(ValueError, match="Unsupported dataset"):
        expected_diagnosis_stage({"fault_stage": "bad"})

    assert expected_detection_label({"label": "supported"}) == "faithful"
    assert expected_detection_label({"label": "unsupported"}) == "hallucinated"
    assert expected_detection_label({"label": "faithful"}) == "faithful"
    assert expected_detection_label({"fault_stage": "no_fault_detected"}) == "faithful"
    assert expected_detection_label({"fault_stage": "generation"}) == "hallucinated"


def test_rag_run_materialization_fault_stage_branches() -> None:
    base = {
        "qid": "q",
        "query": "Question?",
        "gt_answer": "Correct",
        "split": "test",
        "source": "rgb",
        "source_qid": "source",
        "injection": {"noise_text": "Noise text."},
    }

    prompt = rag_run_from_fits_record(
        {**base, "fault_stage": "prompt", "fault_answer": "Wrong prompt."},
        query="Override?",
        seed=1,
    )
    generation = rag_run_from_fits_record({**base, "fault_stage": "generation"}, seed=2)
    no_fault = rag_run_from_fits_record(
        {**base, "fault_stage": "no_fault", "faithful_answer": "Faithful."},
        seed=3,
    )

    assert prompt.query == "Override?"
    assert prompt.answer == "Wrong prompt."
    assert generation.answer == "Correct is not the correct answer."
    assert no_fault.answer == "Faithful."
    assert no_fault.passages == ["Correct. This passage directly supports the answer."]


def test_oracle_mode_runtime_and_model_helpers(tmp_path: Path) -> None:
    assert resolve_oracle_mode({}, default="fixture") == "fixture"
    assert resolve_oracle_mode({"oracle_mode": "real"}) == "real"
    with pytest.raises(ValueError, match="oracle_mode"):
        resolve_oracle_mode({"oracle_mode": "remote"})

    assert resolve_runtime_config_path({"runtime_config": "a.yaml"}) == Path("a.yaml")
    assert resolve_runtime_config_path(
        {}, section={"runtime_config": Path("b.yaml")}
    ) == Path("b.yaml")
    assert resolve_runtime_config_path({}) is None
    with pytest.raises(ValueError, match="runtime_config"):
        resolve_runtime_config_path({"runtime_config": 42})

    assert isinstance(load_runtime_config(None), MutOracleConfig)
    config = MutOracleConfig()
    assert real_model_ids(
        ["nli", "semantic_similarity", "llm_judge", "missing"],
        config=config,
        generation_model="gen",
    ) == [
        "gen",
        config.oracles.nli_model,
        config.oracles.semantic_model,
        config.models.judge,
    ]
    assert fixture_model_ids(["nli", "custom", "nli"]) == [
        "fixture-fits-generator",
        "fixture-nli",
        "fixture-custom",
    ]
    assert (
        provider_route_for_oracles(mode="fixture", oracle_names=["llm_judge"])
        == "fixture"
    )
    assert (
        provider_route_for_oracles(mode="real", oracle_names=["llm_judge"])
        == "openrouter"
    )
    assert (
        provider_route_for_oracles(mode="real", oracle_names=["nli"]) == "local_models"
    )

    ledger_path = tmp_path / "cache.db"
    from mutoracle.cache import SQLiteCacheLedger

    built = real_oracles(
        ["nli", "semantic_similarity", "llm_judge"],
        config=config.model_copy(
            update={
                "runtime": config.runtime.model_copy(update={"cache_path": ledger_path})
            }
        ),
        ledger=SQLiteCacheLedger(ledger_path),
    )
    assert [oracle.name for oracle in built] == [
        "nli",
        "semantic_similarity",
        "llm_judge",
    ]
    with pytest.raises(ValueError, match="Unsupported oracle"):
        real_oracles(
            ["bad"], config=config, ledger=SQLiteCacheLedger(tmp_path / "bad.db")
        )


def test_aggregator_cost_and_full_run_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = ExperimentRunSettings(
        experiment_id="helpers",
        title="helpers",
        mode="full",
        config_path=tmp_path / "config.yaml",
        dataset_path=Path("data/fits/fits_v1.0.0/fits.jsonl"),
        split="test",
        query_limit=1,
        seeds=[13],
        output_dir=tmp_path,
        cost_cap_usd=0.0,
        estimated_cost_per_example_usd=0.0,
        require_smoke_before_full=True,
    )
    paths = artifact_paths(settings)

    assert build_experiment_aggregator(strategy="uniform").combine({"a": 1.0}) == 1.0
    assert (
        build_experiment_aggregator(
            strategy="confidence_gated",
            weights={"nli": 1.0},
            min_score=0.5,
            min_oracles=1,
        ).combine({"nli": 0.8})
        == 0.8
    )
    with pytest.raises(ValueError, match="Unsupported aggregation"):
        build_experiment_aggregator(strategy="bad")

    monkeypatch.setenv("OPENROUTER_DAILY_USD_CAP", "0.01")
    with pytest.raises(RuntimeError, match="Cost gate blocked"):
        enforce_cost_gate(settings, estimated_cost_usd=0.02, confirm_cost=False)
    enforce_cost_gate(settings, estimated_cost_usd=0.02, confirm_cost=True)
    print_cost_estimate(settings, estimated_cost_usd=0.02)
    assert "OPENROUTER_DAILY_USD_CAP" in capsys.readouterr().out

    with pytest.raises(RuntimeError, match="Full run blocked"):
        ensure_full_run_allowed(settings, paths=paths, confirmed_smoke=False)
    ensure_full_run_allowed(settings, paths=paths, confirmed_smoke=True)
    smoke_manifest = paths.manifest_json.with_name("helpers_smoke_manifest.json")
    smoke_manifest.write_text("{}", encoding="utf-8")
    ensure_full_run_allowed(settings, paths=paths, confirmed_smoke=False)

    smoke_settings = settings.__class__(
        **{**settings.__dict__, "mode": "smoke", "require_smoke_before_full": False}
    )
    ensure_full_run_allowed(
        smoke_settings, paths=artifact_paths(smoke_settings), confirmed_smoke=False
    )


def test_mutoracle_detection_baseline_reuses_real_oracles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script("experiments/run_baselines.py")
    call_counter = {"real_oracles": 0}

    class StubOracle:
        def __init__(self, name: str) -> None:
            self.name = name

        def score_result(self, run: object) -> object:
            del run
            return SimpleNamespace(value=1.0, metadata={})

    class StubLocalizer:
        def __init__(
            self,
            *,
            pipeline: object,
            oracles: list[StubOracle],
            aggregator: object,
            delta_threshold: float,
            seed: int,
        ) -> None:
            del pipeline, oracles, aggregator, delta_threshold, seed

        def diagnose(self, query: str) -> object:
            del query
            return SimpleNamespace(
                stage="no_fault_detected",
                confidence=0.1,
                stage_deltas={"prompt": 0.0},
            )

    def fake_real_oracles(
        names: list[str],
        *,
        config: MutOracleConfig,
        ledger: object,
    ) -> list[StubOracle]:
        del config, ledger
        call_counter["real_oracles"] += 1
        return [StubOracle(name) for name in names]

    monkeypatch.setattr(module, "real_oracles", fake_real_oracles)
    monkeypatch.setattr(module, "FaultLocalizer", StubLocalizer)
    monkeypatch.setattr(
        module,
        "fault_report_to_dict",
        lambda report: {"deltas": {}, "stage_deltas": report.stage_deltas},
    )

    baseline = module.MutOracleDetectionBaseline(
        oracle_mode="real",
        runtime_config=MutOracleConfig(),
        ledger=None,
    )
    run = RAGRun(
        query="Who wrote the notes?",
        passages=["Ada wrote the notes."],
        answer="Ada wrote the notes.",
        metadata={"generation": {"model": "fixture-fits-generator", "seed": 13}},
    )

    baseline.run(run)
    baseline.run(run)

    assert call_counter["real_oracles"] == 1


def _load_script(path: str) -> ModuleType:
    spec = util.spec_from_file_location("phase8_script", Path(path))
    assert spec is not None
    assert spec.loader is not None
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
