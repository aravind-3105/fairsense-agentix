"""Utilities for auto-detecting input types and loading content."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast


InputType = Literal["text", "image", "csv"]
WorkflowHint = Literal[
    "bias_text",
    "bias_image",
    "bias_image_vlm",
    "risk",
    "text",
    "image",
    "csv",
]


def detect_input_type(content: object) -> InputType:
    """Infer workflow type from payload."""
    detected: InputType = "text"
    if isinstance(content, (bytes, bytearray)):
        detected = "image"
    elif isinstance(content, Path):
        suffix = content.suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}:
            detected = "image"
        elif suffix == ".csv":
            detected = "csv"
    elif isinstance(content, str) and looks_like_csv(content):
        detected = "csv"

    return detected


def normalize_input_hint(hint: WorkflowHint | None) -> InputType | None:
    """Map workflow IDs to canonical input types."""
    if hint is None:
        return None
    mapping: dict[str, InputType] = {
        "bias_text": "text",
        "bias_image": "image",
        "bias_image_vlm": "image",
        "risk": "csv",
    }
    return mapping.get(hint, cast(InputType, hint))


def load_input_payload(path: Path, input_type: InputType) -> str | bytes:
    """Read file data for orchestrator consumption."""
    if input_type == "image":
        return path.read_bytes()
    return path.read_text(encoding="utf-8")


def looks_like_csv(value: str) -> bool:
    """Heuristic for detecting CSV strings.

    Requires at least 2 lines that all contain commas, and that the comma
    count is consistent across lines (i.e. looks like a table, not prose).
    Single-line strings — even with commas — are treated as plain text.
    """
    if "," not in value:
        return False
    sample = value.strip().splitlines()[:3]
    if len(sample) < 2:
        return False
    if not all("," in line for line in sample):
        return False
    # Comma counts should be consistent across lines (same number of columns)
    counts = [line.count(",") for line in sample]
    return len(set(counts)) == 1
