"""Synthetic regression cases; no original customer records are distributed."""

from pathlib import Path

# Integration fixtures use TemporaryDirectory under this ignored output directory.
(Path(__file__).resolve().parents[1] / "outputs").mkdir(exist_ok=True)
