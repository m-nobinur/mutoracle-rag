"""Dataset manifest and checksum helpers for Phase 6."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DatasetManifest(BaseModel):
    """Source dataset provenance recorded at data build time."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    name: str
    url: str = Field(pattern=r"^https?://")
    license: str
    revision: str
    checksum: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    date: str
    notes: str


def sha256_bytes(content: bytes) -> str:
    """Return a plan-compatible SHA-256 digest string."""

    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def sha256_text(content: str) -> str:
    """Return a SHA-256 digest for UTF-8 text."""

    return sha256_bytes(content.encode("utf-8"))


def sha256_file(path: Path) -> str:
    """Return a SHA-256 digest for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def json_dump(data: Any, path: Path) -> None:
    """Write stable indented JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
