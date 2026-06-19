"""Shared public analysis paths and config loading."""

from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_OUTPUT_ROOT = REPO_ROOT / "analysis" / "outputs"


def load_yaml_config(path: str | Path) -> dict:
    """Load a YAML analysis config."""
    with Path(path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config or {}


def resolve_path(path: str | Path, *, base: Path) -> Path:
    """Resolve a path relative to a base directory."""
    path = Path(path)
    return path if path.is_absolute() else (base / path).resolve()


def visualization_output_dir(
    output_root: str | Path,
    config: dict,
    script_file: str | Path,
) -> Path:
    """Return a visualization output directory scoped by representation and script."""
    return (
        Path(output_root)
        / config["evaluation"]["output_prefix"]
        / Path(script_file).stem
    )
