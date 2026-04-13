"""Map bias analysis instances to character spans in merged OCR/caption text."""

from fairsense_agentix.services.telemetry import telemetry
from fairsense_agentix.tools.llm.output_schemas import BiasAnalysisOutput


def _extract_spans_from_analysis(  # noqa: PLR0912, PLR0915
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

    # Extract spans with validation, fallback logic, and offset adjustment
    spans: list[tuple[int, int, str]] = []
    merged_length = len(merged_text)

    for idx, instance in enumerate(analysis.bias_instances):
        # Determine which text source this span refers to
        evidence_source = instance.evidence_source

        if evidence_source == "ocr_text":
            source_text = ocr_text
            offset = ocr_offset
        elif evidence_source == "caption":
            source_text = caption_text
            offset = caption_offset
        else:
            # Unknown source or "text" (generic) - skip
            telemetry.log_warning(
                "image_span_unknown_source",
                instance_idx=idx,
                evidence_source=evidence_source,
            )
            continue

        # Get positions relative to source text
        start = instance.start_char
        end = instance.end_char

        # Debug logging for position values
        telemetry.log_info(
            "image_extract_span_attempt",
            instance_idx=idx,
            text_span=instance.text_span[:30] if instance.text_span else None,
            start_char=start,
            end_char=end,
            evidence_source=evidence_source,
            bias_type=instance.type,
        )

        # FALLBACK: If LLM didn't provide positions or they're invalid,
        # calculate from text_span
        needs_fallback = False

        # Check if positions are missing (defaulted to 0, 0)
        if start == 0 and end == 0:
            telemetry.log_warning(
                "image_span_positions_missing",
                instance_idx=idx,
                text_span=instance.text_span,
                evidence_source=evidence_source,
            )
            needs_fallback = True
        # Check if positions are out of bounds in source text
        elif not (0 <= start < len(source_text) and start < end <= len(source_text)):
            telemetry.log_warning(
                "image_span_positions_out_of_bounds",
                instance_idx=idx,
                start=start,
                end=end,
                source_length=len(source_text),
                evidence_source=evidence_source,
            )
            needs_fallback = True
        # Check if text_span doesn't match the positions in source text
        elif instance.text_span and source_text[start:end] != instance.text_span:
            actual_text = source_text[start:end]
            telemetry.log_warning(
                "image_span_text_mismatch",
                instance_idx=idx,
                expected=instance.text_span,
                actual=actual_text,
                positions=f"{start}-{end}",
                evidence_source=evidence_source,
            )
            needs_fallback = True

        # Try fallback calculation if needed
        if needs_fallback:
            if instance.text_span:
                # Try to find the text span in the source text
                found_pos = source_text.find(instance.text_span)
                if found_pos >= 0:
                    start = found_pos
                    end = found_pos + len(instance.text_span)
                    telemetry.log_info(
                        "image_span_fallback_success",
                        instance_idx=idx,
                        text_span=instance.text_span,
                        calculated_start=start,
                        calculated_end=end,
                        evidence_source=evidence_source,
                    )
                else:
                    # Try case-insensitive search as last resort
                    source_lower = source_text.lower()
                    span_lower = instance.text_span.lower()
                    found_pos = source_lower.find(span_lower)
                    if found_pos >= 0:
                        start = found_pos
                        end = found_pos + len(instance.text_span)
                        telemetry.log_info(
                            "image_span_fallback_case_insensitive",
                            instance_idx=idx,
                            text_span=instance.text_span,
                            calculated_start=start,
                            calculated_end=end,
                            evidence_source=evidence_source,
                        )
                    else:
                        telemetry.log_error(
                            "image_span_fallback_failed",
                            instance_idx=idx,
                            text_span=instance.text_span,
                            evidence_source=evidence_source,
                            reason="text_span not found in source text",
                        )
                        continue
            else:
                telemetry.log_error(
                    "image_span_fallback_impossible",
                    instance_idx=idx,
                    evidence_source=evidence_source,
                    reason="no text_span provided",
                )
                continue

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
