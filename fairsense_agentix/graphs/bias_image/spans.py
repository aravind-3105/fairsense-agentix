"""Map bias analysis instances to character spans in merged OCR/caption text."""

from fairsense_agentix.services.telemetry import telemetry
from fairsense_agentix.tools.llm.output_schemas import BiasAnalysisOutput, BiasInstance


def _resolve_source(  # noqa: PLR0913
    evidence_source: str,
    ocr_text: str,
    caption_text: str,
    ocr_offset: int,
    caption_offset: int,
    idx: int,
) -> tuple[str, int] | None:
    """Return (source_text, offset) for the given evidence_source.

    Returns None and logs a warning if the source is unrecognised.
    """
    if evidence_source == "ocr_text":
        return ocr_text, ocr_offset
    if evidence_source == "caption":
        return caption_text, caption_offset
    # Unknown source or "text" (generic) - skip
    telemetry.log_warning(
        "image_span_unknown_source",
        instance_idx=idx,
        evidence_source=evidence_source,
    )
    return None


def _needs_fallback(instance: BiasInstance, source_text: str, idx: int) -> bool:
    """Check whether the LLM-provided character positions need a fallback.

    Validates three conditions and logs the appropriate warning for each:
    - positions are missing (both defaulted to 0)
    - positions are out of bounds in the source text
    - text at positions does not match instance.text_span
    """
    start = instance.start_char
    end = instance.end_char

    if start == 0 and end == 0:
        telemetry.log_warning(
            "image_span_positions_missing",
            instance_idx=idx,
            text_span=instance.text_span,
            evidence_source=instance.evidence_source,
        )
        return True

    if not (0 <= start < len(source_text) and start < end <= len(source_text)):
        telemetry.log_warning(
            "image_span_positions_out_of_bounds",
            instance_idx=idx,
            start=start,
            end=end,
            source_length=len(source_text),
            evidence_source=instance.evidence_source,
        )
        return True

    if instance.text_span and source_text[start:end] != instance.text_span:
        telemetry.log_warning(
            "image_span_text_mismatch",
            instance_idx=idx,
            expected=instance.text_span,
            actual=source_text[start:end],
            positions=f"{start}-{end}",
            evidence_source=instance.evidence_source,
        )
        return True

    return False


def _fallback_span(
    instance: BiasInstance,
    source_text: str,
    idx: int,
) -> tuple[int, int] | None:
    """Resolve (start, end) for an instance when LLM positions are invalid.

    Tries an exact search first, then a case-insensitive search as last resort.
    Returns None if the text_span cannot be located.
    """
    if not instance.text_span:
        telemetry.log_error(
            "image_span_fallback_impossible",
            instance_idx=idx,
            evidence_source=instance.evidence_source,
            reason="no text_span provided",
        )
        return None

    found_pos = source_text.find(instance.text_span)
    if found_pos >= 0:
        start, end = found_pos, found_pos + len(instance.text_span)
        telemetry.log_info(
            "image_span_fallback_success",
            instance_idx=idx,
            text_span=instance.text_span,
            calculated_start=start,
            calculated_end=end,
            evidence_source=instance.evidence_source,
        )
        return start, end

    # Try case-insensitive search as last resort
    found_pos = source_text.lower().find(instance.text_span.lower())
    if found_pos >= 0:
        start, end = found_pos, found_pos + len(instance.text_span)
        telemetry.log_info(
            "image_span_fallback_case_insensitive",
            instance_idx=idx,
            text_span=instance.text_span,
            calculated_start=start,
            calculated_end=end,
            evidence_source=instance.evidence_source,
        )
        return start, end

    telemetry.log_error(
        "image_span_fallback_failed",
        instance_idx=idx,
        text_span=instance.text_span,
        evidence_source=instance.evidence_source,
        reason="text_span not found in source text",
    )
    return None


def _resolve_instance_span(
    instance: BiasInstance,
    idx: int,
    source_text: str,
) -> tuple[int, int] | None:
    """Return (start, end) positions relative to source_text, or None to skip.

    Logs the attempt, validates positions, and applies fallback logic when needed.
    """
    start = instance.start_char
    end = instance.end_char

    telemetry.log_info(
        "image_extract_span_attempt",
        instance_idx=idx,
        text_span=instance.text_span[:30] if instance.text_span else None,
        start_char=start,
        end_char=end,
        evidence_source=instance.evidence_source,
        bias_type=instance.type,
    )

    if _needs_fallback(instance, source_text, idx):
        result = _fallback_span(instance, source_text, idx)
        if result is None:
            return None
        start, end = result

    return start, end


def _extract_spans_from_analysis(
    bias_analysis: str | BiasAnalysisOutput,
    merged_text: str,
    ocr_text: str,
    caption_text: str,
) -> list[tuple[int, int, str]]:
    """Extract bias spans with character positions from image analysis.

    Phase 6.2: Parse BiasAnalysisOutput to extract (start_char, end_char, type)
    tuples for HTML highlighting. Handles multiple text sources (OCR + caption)
    with evidence_source tracking.

    Phase 7.1: Added fallback logic to calculate positions from text_span when
    LLM doesn't provide accurate character positions or positions are invalid.

    For image analysis, merged_text combines OCR and caption with headers.
    We map character positions from individual sources to merged_text.

    Parameters
    ----------
    bias_analysis : str | BiasAnalysisOutput
        Bias analysis output (JSON string or Pydantic object)
    merged_text : str
        Combined OCR + caption text that was analyzed
    ocr_text : str
        Original OCR extracted text
    caption_text : str
        Original caption generated text

    Returns
    -------
    list[tuple[int, int, str]]
        List of (start_char, end_char, bias_type) tuples for highlighting
        in the merged_text

    Examples
    --------
    >>> analysis = BiasAnalysisOutput(
    ...     bias_detected=True,
    ...     bias_instances=[
    ...         BiasInstance(
    ...             type="gender",
    ...             evidence_source="ocr_text",
    ...             start_char=0,
    ...             end_char=8,
    ...         )
    ...     ],
    ... )
    >>> _extract_spans_from_analysis(analysis, merged, "ocr", "caption")
    [(27, 35, 'gender')]  # Adjusted for merged_text offset
    """
    # Parse analysis if string
    if isinstance(bias_analysis, str):
        try:
            analysis = BiasAnalysisOutput.model_validate_json(bias_analysis)
        except Exception:
            return []
    else:
        analysis = bias_analysis

    # Calculate offsets for OCR and caption in merged_text
    # Format: "**OCR Extracted Text:**\n{ocr}\n\n**Image Caption:**\n{caption}"
    ocr_header = "**OCR Extracted Text:**\n"
    caption_header = "\n\n**Image Caption:**\n"
    ocr_offset = len(ocr_header)
    caption_offset = len(ocr_header) + len(ocr_text) + len(caption_header)

    spans: list[tuple[int, int, str]] = []
    merged_length = len(merged_text)

    for idx, instance in enumerate(analysis.bias_instances):
        resolved = _resolve_source(
            instance.evidence_source,
            ocr_text,
            caption_text,
            ocr_offset,
            caption_offset,
            idx,
        )
        if resolved is None:
            continue
        source_text, offset = resolved

        span = _resolve_instance_span(instance, idx, source_text)
        if span is None:
            continue
        start, end = span

        # Adjust to merged_text coordinates
        merged_start = offset + start
        merged_end = offset + end

        # Final validation in merged_text
        if not (
            0 <= merged_start < merged_length
            and merged_start < merged_end <= merged_length
        ):
            telemetry.log_error(
                "image_span_merged_validation_failed",
                instance_idx=idx,
                merged_start=merged_start,
                merged_end=merged_end,
                merged_length=merged_length,
            )
            continue

        # Valid span - add to list
        spans.append((merged_start, merged_end, instance.type))
        telemetry.log_info(
            "image_span_added",
            instance_idx=idx,
            merged_start=merged_start,
            merged_end=merged_end,
            type=instance.type,
            text_preview=merged_text[merged_start:merged_end][:30],
        )

    telemetry.log_info(
        "image_extract_spans_complete",
        total_instances=len(analysis.bias_instances),
        extracted_spans=len(spans),
        filtered_out=len(analysis.bias_instances) - len(spans),
    )

    return spans
