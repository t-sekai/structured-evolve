"""Candidate metadata for Level-1 schedule evolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Candidate:
    """One generated schedule candidate in an evolution run."""

    candidate_id: str
    generation: int
    path: Path
    parent_id: str | None = None
    prompt_path: Path | None = None
    response_path: Path | None = None

