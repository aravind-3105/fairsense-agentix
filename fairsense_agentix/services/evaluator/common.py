"""Shared types used by both bias and risk evaluators."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvaluationContext:
    """Container describing what was evaluated (for metadata)."""

    workflow_id: str
    run_id: str | None
