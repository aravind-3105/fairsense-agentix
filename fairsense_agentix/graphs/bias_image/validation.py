"""Validate image payloads before OCR/caption nodes run."""

import io

from PIL import Image, UnidentifiedImageError

from fairsense_agentix.configs import settings
from fairsense_agentix.graphs.state import BiasImageState
from fairsense_agentix.services.telemetry import telemetry


def _ensure_valid_image_bytes(state: BiasImageState) -> None:
    """Validate that provided image bytes represent a decodable image.

    When fake OCR/caption tools are configured they happily accept any bytes,
    which can mask corrupted inputs and breaks stress tests that expect early
    failures. We proactively run a lightweight Pillow validation so invalid
    payloads raise ValueError regardless of which tools are configured.
    """
    should_validate = state.options.get(
        "validate_image_bytes",
        settings.image_validation_enabled,
    )
    if not should_validate:
        return

    # Avoid double validation when OCR and caption nodes both run.
    if state.options.get("_image_validation_passed"):
        return

    if not state.image_bytes:
        telemetry.log_error("bias_image_invalid_bytes", reason="empty_payload")
        raise ValueError("Image data is empty; provide a valid image file.")

    try:
        with Image.open(io.BytesIO(state.image_bytes)) as img:
            img.verify()
    except UnidentifiedImageError as exc:
        telemetry.log_error(
            "bias_image_invalid_bytes",
            reason="unidentified",
            error=exc,
        )
        raise ValueError("Invalid image data: unable to decode image bytes.") from exc
    except Exception as exc:
        telemetry.log_error(
            "bias_image_invalid_bytes",
            reason="decode_failed",
            error=exc,
        )
        raise ValueError("Invalid or corrupted image data.") from exc

    # Mark as validated so parallel branch skips redundant work.
    state.options["_image_validation_passed"] = True
