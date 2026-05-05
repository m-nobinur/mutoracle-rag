from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def test_weight_search_is_reproducible_for_fixed_seed() -> None:
    module = _load_weight_search_module()

    first = module.run_weight_search(seed=2026)
    second = module.run_weight_search(seed=2026)

    assert first == second
    assert first["accuracy"] == 1.0
    assert first["delta_threshold"] > 0.0


def test_weight_search_uses_seed_for_tie_breaking(monkeypatch) -> None:
    module = _load_weight_search_module()

    tied_weights = [
        {"nli": 0.5, "semantic_similarity": 0.5, "llm_judge": 0.0},
        {"nli": 0.5, "semantic_similarity": 0.0, "llm_judge": 0.5},
    ]
    base_scores = {"nli": 1.0, "semantic_similarity": 1.0, "llm_judge": 1.0}
    mutated_scores = {
        operator_id: dict(base_scores) for operator_id in module.OPERATOR_STAGES
    }
    tied_example = module.CalibrationExample(
        label="no_fault_detected",
        baseline=dict(base_scores),
        mutated=mutated_scores,
    )

    monkeypatch.setattr(
        module,
        "candidate_weights",
        lambda: iter([dict(weight) for weight in tied_weights]),
    )
    monkeypatch.setattr(module, "THRESHOLD_CANDIDATES", (0.05,))
    monkeypatch.setattr(module, "CALIBRATION_EXAMPLES", (tied_example,))

    selected_weights = {
        tuple(sorted(module.run_weight_search(seed=seed)["weights"].items()))
        for seed in range(8)
    }

    assert len(selected_weights) > 1


def _load_weight_search_module() -> ModuleType:
    path = Path("experiments/run_weight_search.py")
    spec = importlib.util.spec_from_file_location("run_weight_search", path)
    if spec is None or spec.loader is None:
        msg = "Unable to load experiments/run_weight_search.py"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
