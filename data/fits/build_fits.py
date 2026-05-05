"""Script entry point for the Phase 6 FITS builder."""

from __future__ import annotations

import argparse
from pathlib import Path

from mutoracle.data import build_fits_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FITS v1.0.0 artifacts.")
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--version", default="fits_v1.0.0")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild an existing FITS artifact directory.",
    )
    args = parser.parse_args()
    paths = build_fits_dataset(
        output_root=args.output_root,
        seed=args.seed,
        version=args.version,
        force_rebuild=args.force,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
