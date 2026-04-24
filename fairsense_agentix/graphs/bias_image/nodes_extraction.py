"""Bias image graph nodes: OCR, caption, and merge."""

from fairsense_agentix.graphs.bias_image.validation import _ensure_valid_image_bytes
from fairsense_agentix.graphs.state import BiasImageState
from fairsense_agentix.services.telemetry import telemetry
from fairsense_agentix.tools import get_tool_registry


def extract_ocr(state: BiasImageState) -> dict[str, str]:
    """OCR node: Extract text from image using OCR tool.

    Uses OCR (Optical Character Recognition) to extract text content from
    the image. Supports multiple OCR backends (Tesseract, PaddleOCR, etc.).

    Executes in parallel with generate_caption for faster processing.

    Phase 4+: Uses tool registry to get OCR implementation (fake or real).
    Phase 5+: Real OCR integration with confidence thresholds.
    Phase 6.4: Added telemetry for observability and performance tracking.

    Parameters
    ----------
    state : BiasImageState
        Current state with image_bytes

    Returns
    -------
    dict
        State update with ocr_text field

    Examples
    --------
    >>> state = BiasImageState(image_bytes=b"image_data")
    >>> update = extract_ocr(state)
    >>> "ocr_text" in update
    True
    """
    import logging
    import time

    logger = logging.getLogger(__name__)
    node_start = time.time()

    with telemetry.timer("bias_image.extract_ocr", image_size=len(state.image_bytes)):
        logger.info(
            f"👁️  [IMAGE GRAPH] Starting OCR extraction "
            f"(image_size={len(state.image_bytes)} bytes)",
        )
        telemetry.log_info(
            "bias_image_ocr_start",
            image_size=len(state.image_bytes),
            run_id=state.run_id or "unknown",
        )

        try:
            _ensure_valid_image_bytes(state)
            # Get tools from registry
            registry = get_tool_registry()

            # Extract options
            language = state.options.get("ocr_language", "eng")
            confidence_threshold = state.options.get("ocr_confidence", 0.5)

            # Validate parameters
            if not isinstance(confidence_threshold, (int, float)) or not (
                0.0 <= confidence_threshold <= 1.0
            ):
                telemetry.log_warning(
                    "invalid_ocr_confidence",
                    confidence_threshold=confidence_threshold,
                    using_default=0.5,
                )
                confidence_threshold = 0.5  # Use default for invalid values
            if not isinstance(language, str) or not language:
                telemetry.log_warning(
                    "invalid_ocr_language",
                    language=language,
                    using_default="eng",
                )
                language = "eng"  # Use default for invalid values

            # Use OCR tool from registry with error handling
            try:
                with telemetry.timer("bias_image.ocr_tool", language=language):
                    ocr_text = registry.ocr.extract(
                        image_bytes=state.image_bytes,
                        language=language,
                        confidence_threshold=confidence_threshold,
                    )
                telemetry.log_info("ocr_extraction_complete", text_length=len(ocr_text))
            except Exception as e:
                telemetry.log_error("ocr_extraction_failed", error=e)
                # Fallback: Empty string allows workflow to continue with caption-only
                # OCR can fail for many reasons (bad image format, corrupted data)
                # Since caption runs in parallel, continue with single data source
                ocr_text = ""

            node_time = time.time() - node_start
            logger.info(
                f"✅ [IMAGE GRAPH] OCR node complete in {node_time:.2f}s "
                f"(text_length={len(ocr_text)})",
            )
            return {"ocr_text": ocr_text}

        except Exception as e:
            telemetry.log_error("bias_image_extract_ocr_failed", error=e)
            raise


def generate_caption(state: BiasImageState) -> dict[str, str]:
    """Captioning node: Generate descriptive caption for image.

    Uses image captioning models (BLIP, BLIP-2, LLaVA, etc.) to generate
    a natural language description of the image content. This captures
    visual elements that OCR might miss.

    Executes in parallel with extract_ocr for faster processing.

    Phase 4+: Uses tool registry to get caption implementation (fake or real).
    Phase 5+: Real captioning using BLIP-2 or similar models.
    Phase 6.4: Added telemetry for observability and performance tracking.

    Parameters
    ----------
    state : BiasImageState
        Current state with image_bytes

    Returns
    -------
    dict
        State update with caption_text field

    Examples
    --------
    >>> state = BiasImageState(image_bytes=b"image_data")
    >>> update = generate_caption(state)
    >>> "caption_text" in update
    True
    """
    import logging
    import time

    logger = logging.getLogger(__name__)
    node_start = time.time()

    with telemetry.timer(
        "bias_image.generate_caption",
        image_size=len(state.image_bytes),
    ):
        logger.info(
            f"🖼️  [IMAGE GRAPH] Starting caption generation "
            f"(image_size={len(state.image_bytes)} bytes)",
        )
        telemetry.log_info(
            "bias_image_caption_start",
            image_size=len(state.image_bytes),
            run_id=state.run_id or "unknown",
        )

        try:
            _ensure_valid_image_bytes(state)
            # Get tools from registry
            registry = get_tool_registry()

            # Extract options
            max_length = state.options.get("caption_max_length", 100)

            # Validate max_length parameter
            if not isinstance(max_length, int) or max_length <= 0:
                telemetry.log_warning(
                    "invalid_caption_max_length",
                    max_length=max_length,
                    using_default=100,
                )
                max_length = 100  # Use default for invalid values

            # Use caption tool from registry with error handling
            try:
                with telemetry.timer("bias_image.caption_tool", max_length=max_length):
                    caption_text = registry.caption.caption(
                        image_bytes=state.image_bytes,
                        max_length=max_length,
                    )
                telemetry.log_info(
                    "caption_generation_complete",
                    text_length=len(caption_text),
                )
            except Exception as e:
                telemetry.log_error("caption_generation_failed", error=e)
                # Fallback: Empty string allows workflow to continue with OCR-only
                # Caption models can fail (model loading error, OOM, invalid image)
                # Since OCR runs in parallel, continue with single data source
                caption_text = ""

            node_time = time.time() - node_start
            logger.info(
                f"✅ [IMAGE GRAPH] Caption node complete in {node_time:.2f}s "
                f"(caption_length={len(caption_text)})",
            )
            return {"caption_text": caption_text}

        except Exception as e:
            telemetry.log_error("bias_image_generate_caption_failed", error=e)
            raise


def merge_text(state: BiasImageState) -> dict[str, str]:
    """Merge node: Combine OCR and caption text into unified input.

    Merges the extracted OCR text and generated caption into a single
    text input for bias analysis. Performs deduplication to avoid
    redundancy.

    Phase 6.4: Added telemetry for observability and performance tracking.

    Parameters
    ----------
    state : BiasImageState
        Current state with ocr_text and caption_text

    Returns
    -------
    dict
        State update with merged_text field

    Examples
    --------
    >>> state = BiasImageState(
    ...     image_bytes=b"...",
    ...     ocr_text="Text from OCR",
    ...     caption_text="Description from caption",
    ... )
    >>> update = merge_text(state)
    >>> "merged_text" in update
    True
    """
    with telemetry.timer("bias_image.merge_text"):
        telemetry.log_info(
            "bias_image_merge_start",
            run_id=state.run_id or "unknown",
            has_ocr=bool(state.ocr_text),
            has_caption=bool(state.caption_text),
        )

        try:
            # Phase 2: Simple concatenation
            # Phase 5+: Smart merging with deduplication and formatting

            ocr_text = state.ocr_text or ""
            caption_text = state.caption_text or ""

            # If BOTH extraction methods failed, we can't continue
            if not ocr_text and not caption_text:
                telemetry.log_error(
                    "dual_extraction_failed",
                    ocr_empty=not ocr_text,
                    caption_empty=not caption_text,
                )
                # Raise error that orchestrator will catch and handle
                raise ValueError(
                    "Both OCR and caption extraction failed - no text available "
                    "for analysis",
                )

            # Simple merge with section headers
            merged = f"""**OCR Extracted Text:**
{ocr_text}

**Image Caption:**
{caption_text}"""

            telemetry.log_info(
                "bias_image_merge_complete",
                merged_length=len(merged),
                ocr_length=len(ocr_text),
                caption_length=len(caption_text),
            )
            return {"merged_text": merged}

        except Exception as e:
            telemetry.log_error("bias_image_merge_text_failed", error=e)
            raise
