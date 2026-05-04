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
