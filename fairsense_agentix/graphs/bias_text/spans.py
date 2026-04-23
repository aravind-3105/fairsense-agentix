"""Map bias analysis instances to character spans in source text."""

from fairsense_agentix.services.telemetry import telemetry
from fairsense_agentix.tools.llm.output_schemas import BiasAnalysisOutput


def _extract_spans_from_analysis(
    bias_analysis: str | BiasAnalysisOutput,
    original_text: str,
) -> list[tuple[int, int, str]]:
    """Extract bias spans with character positions from analysis.

    Phase 6.1: Parse BiasAnalysisOutput to extract (start_char, end_char, type)
    tuples for HTML highlighting. Validates positions are within text bounds.

    Phase 7.1: Added fallback logic to calculate positions from text_span when
    LLM doesn't provide accurate character positions or positions are invalid.

    Parameters
    ----------
    bias_analysis : str | BiasAnalysisOutput
        Bias analysis output (JSON string or Pydantic object)
    original_text : str
        Original text being analyzed (for bounds checking)

    Returns
    -------
    list[tuple[int, int, str]]
        List of (start_char, end_char, bias_type) tuples for highlighting

    Examples
    --------
    >>> analysis = BiasAnalysisOutput(
    ...     bias_detected=True,
    ...     bias_instances=[
    ...         BiasInstance(
    ...             type="gender",
    ...             severity="high",
    ...             text_span="salesman",
    ...             explanation="Gendered term",
    ...             start_char=0,
    ...             end_char=8,
    ...         )
    ...     ],
    ...     overall_assessment="Gender bias detected",
    ...     risk_level="high",
    ... )
    >>> _extract_spans_from_analysis(analysis, "salesman needed")
    [(0, 8, 'gender')]
    """
    # bias_analysis should always be BiasAnalysisOutput now (not string)
    if not isinstance(bias_analysis, BiasAnalysisOutput):
        # This shouldn't happen with the new design
        telemetry.log_warning("extract_spans_invalid_type", type=type(bias_analysis))
        return []

    analysis = bias_analysis

    # Extract spans with validation and fallback logic
    spans: list[tuple[int, int, str]] = []
    text_length = len(original_text)

    for idx, instance in enumerate(analysis.bias_instances):
        start = instance.start_char
        end = instance.end_char

        # Debug logging for position values
        telemetry.log_info(
            "extract_span_attempt",
            instance_idx=idx,
            text_span=instance.text_span[:30] if instance.text_span else None,
            start_char=start,
            end_char=end,
            bias_type=instance.type,
        )

        # FALLBACK: If LLM didn't provide positions or they're invalid,
        # calculate from text_span
        needs_fallback = False

        # Check if positions are missing (defaulted to 0, 0)
        if start == 0 and end == 0:
            telemetry.log_warning(
                "span_positions_missing",
                instance_idx=idx,
                text_span=instance.text_span,
            )
            needs_fallback = True
        # Check if positions are out of bounds
        elif not (0 <= start < text_length and start < end <= text_length):
            telemetry.log_warning(
                "span_positions_out_of_bounds",
                instance_idx=idx,
                start=start,
                end=end,
                text_length=text_length,
            )
            needs_fallback = True
        # Check if text_span doesn't match the positions
        elif instance.text_span and original_text[start:end] != instance.text_span:
            actual_text = original_text[start:end]
            telemetry.log_warning(
                "span_text_mismatch",
                instance_idx=idx,
                expected=instance.text_span,
                actual=actual_text,
                positions=f"{start}-{end}",
            )
            needs_fallback = True

        # Try fallback calculation if needed
        if needs_fallback:
            if instance.text_span:
                # Try to find the text span in the original text
                found_pos = original_text.find(instance.text_span)
                if found_pos >= 0:
                    start = found_pos
                    end = found_pos + len(instance.text_span)
                    telemetry.log_info(
                        "span_fallback_success",
                        instance_idx=idx,
                        text_span=instance.text_span,
                        calculated_start=start,
                        calculated_end=end,
                    )
                else:
                    # Try case-insensitive search as last resort
                    text_lower = original_text.lower()
                    span_lower = instance.text_span.lower()
                    found_pos = text_lower.find(span_lower)
                    if found_pos >= 0:
                        start = found_pos
                        end = found_pos + len(instance.text_span)
                        telemetry.log_info(
                            "span_fallback_case_insensitive",
                            instance_idx=idx,
                            text_span=instance.text_span,
                            calculated_start=start,
                            calculated_end=end,
                        )
                    else:
                        telemetry.log_error(
                            "span_fallback_failed",
                            instance_idx=idx,
                            text_span=instance.text_span,
                            reason="text_span not found in original text",
                        )
                        continue
            else:
                telemetry.log_error(
                    "span_fallback_impossible",
                    instance_idx=idx,
                    reason="no text_span provided",
                )
                continue

        # Valid span - add to list
        spans.append((start, end, instance.type))
        telemetry.log_info(
            "span_added",
            instance_idx=idx,
            start=start,
            end=end,
            type=instance.type,
            text_preview=original_text[start:end][:30],
        )

    telemetry.log_info(
        "extract_spans_complete",
        total_instances=len(analysis.bias_instances),
        extracted_spans=len(spans),
        filtered_out=len(analysis.bias_instances) - len(spans),
    )

    return spans
