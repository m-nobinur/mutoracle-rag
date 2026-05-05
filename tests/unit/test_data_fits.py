from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

import mutoracle.cli as cli_module
from mutoracle.data import (
    FITSRecord,
    build_fits_dataset,
    build_noise_pool,
    load_rgb_subset,
    load_triviaqa_subset,
    sha256_file,
)
from mutoracle.data.loaders import dataset_manifests


def test_dataset_manifests_include_required_provenance() -> None:
    manifests = dataset_manifests(build_date="2026-05-05")

    assert {manifest.dataset_id for manifest in manifests} == {
        "rgb",
        "triviaqa",
        "wikipedia_noise",
    }
    for manifest in manifests:
        assert str(manifest.url).startswith("https://")
        assert manifest.license
        assert manifest.revision
        assert manifest.checksum.startswith("sha256:")
        assert manifest.date == "2026-05-05"


def test_source_loaders_return_schema_compatible_examples() -> None:
    rgb = load_rgb_subset(limit=3)
    trivia = load_triviaqa_subset(limit=3)
    noise = build_noise_pool(limit=3)

    assert len(rgb) == 3
    assert len(trivia) == 3
    assert len(noise) == 3
    assert rgb[0].source == "rgb"
    assert trivia[0].source == "triviaqa"
    assert {"doc_id", "text"} <= set(noise[0])


def test_build_fits_is_stratified_and_split_safe(tmp_path: Path) -> None:
    paths = build_fits_dataset(output_root=tmp_path, seed=2026)
    records = _read_records(paths["fits"])

    report = json.loads(paths["quality_report"].read_text(encoding="utf-8"))

    assert report["passed"]
    assert report["labels"] == {
        "retrieval": 75,
        "prompt": 75,
        "generation": 75,
        "no_fault": 75,
    }
    assert report["query_length_within_tolerance"] is True
    validation_qids = {record.qid for record in records if record.split == "validation"}
    test_qids = {record.qid for record in records if record.split == "test"}
    assert validation_qids.isdisjoint(test_qids)
    assert len(validation_qids) == 60
    assert len(test_qids) == 240


def test_build_fits_is_deterministic_for_fixed_seed(tmp_path: Path) -> None:
    first = build_fits_dataset(output_root=tmp_path / "first", seed=42)
    second = build_fits_dataset(output_root=tmp_path / "second", seed=42)

    assert sha256_file(first["fits"]) == sha256_file(second["fits"])
    assert sha256_file(first["validation"]) == sha256_file(second["validation"])
    assert sha256_file(first["test"]) == sha256_file(second["test"])


def test_build_fits_reuses_existing_frozen_artifact(tmp_path: Path) -> None:
    first = build_fits_dataset(output_root=tmp_path, seed=2026)
    mtime_before = first["fits"].stat().st_mtime_ns

    second = build_fits_dataset(output_root=tmp_path, seed=2026)

    assert second == first
    assert second["fits"].stat().st_mtime_ns == mtime_before


def test_build_fits_force_rebuild_restores_modified_artifact(tmp_path: Path) -> None:
    paths = build_fits_dataset(output_root=tmp_path, seed=2026)
    fits_path = paths["fits"]
    original_checksum = sha256_file(fits_path)

    fits_path.write_text('{"broken": true}\n', encoding="utf-8")
    assert sha256_file(fits_path) != original_checksum

    rebuilt = build_fits_dataset(output_root=tmp_path, seed=2026, force_rebuild=True)
    assert rebuilt == paths
    assert sha256_file(rebuilt["fits"]) == original_checksum


def test_cli_data_build(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli_module.app,
        ["data", "build", "--output-root", str(tmp_path), "--seed", "2026"],
    )

    assert result.exit_code == 0
    assert "FITS build passed" in result.output
    assert (tmp_path / "fits" / "manifest.json").exists()


def test_cli_fits_build_alias(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli_module.app,
        ["fits", "build", "--output-root", str(tmp_path), "--seed", "2026"],
    )

    assert result.exit_code == 0
    assert "FITS build passed" in result.output


def test_cli_fits_build_force_rebuild(tmp_path: Path) -> None:
    first = CliRunner().invoke(
        cli_module.app,
        ["fits", "build", "--output-root", str(tmp_path), "--seed", "2026"],
    )
    assert first.exit_code == 0

    fits_path = tmp_path / "fits" / "fits_v1.0.0" / "fits.jsonl"
    fits_path.write_text('{"broken": true}\n', encoding="utf-8")

    second = CliRunner().invoke(
        cli_module.app,
        [
            "fits",
            "build",
            "--output-root",
            str(tmp_path),
            "--seed",
            "2026",
            "--force",
        ],
    )
    assert second.exit_code == 0
    assert "FITS build passed" in second.output


def _read_records(path: Path) -> list[FITSRecord]:
    return [
        FITSRecord.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
