"""BiasImageGraph subgraph for image bias analysis.

This workflow extracts information from images via parallel OCR and captioning,
merges the extracted text, then analyzes for bias. The parallel extraction
pattern enables faster processing.

Workflow:
    START → [extract_ocr ∥ generate_caption] → merge_text → analyze_bias
         → summarize → highlight → END

Phase 2 (Checkpoint 2.5): Stub implementations with fake OCR/caption/LLM
Phase 5+: Real OCR integration, captioning models, LLM analysis
Phase 6.2: Span extraction with evidence source tracking (caption vs OCR)

Example:
    >>> from fairsense_agentix.graphs.bias_image_graph import create_bias_image_graph
    >>> graph = create_bias_image_graph()
    >>> result = graph.invoke({"image_bytes": b"fake_image_data", "options": {}})
    >>> result["bias_analysis"]
    '...'
"""

import io
from pathlib import Path

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from PIL import Image, UnidentifiedImageError

from fairsense_agentix.configs import settings
from fairsense_agentix.graphs.state import BiasImageState
from fairsense_agentix.services.telemetry import telemetry
from fairsense_agentix.tools import get_tool_registry
from fairsense_agentix.tools.llm.output_schemas import BiasAnalysisOutput


# ============================================================================
# Helper Functions
# ============================================================================


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


# ============================================================================
# Node Functions
# ============================================================================


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


def analyze_bias(state: BiasImageState) -> dict:
    """LLM node: Analyze merged text for various forms of bias.

    Uses LLM to identify and explain bias in the merged OCR + caption text
    across multiple dimensions (gender, age, race, disability, socioeconomic).

    This is conceptually similar to BiasTextGraph's analyze_bias, but operates
    on the merged text from image analysis.

    Phase 4+: Uses tool registry to get LLM implementation (fake or real).
    Phase 5+: Real LLM call with bias detection prompt.
    Phase 6.2: Uses bias_analysis_v1.txt prompt template with structured output
               and evidence_source tracking (caption vs ocr_text).
    Phase 6.4: Added telemetry for observability and performance tracking.

    Parameters
    ----------
    state : BiasImageState
        Current state with merged_text

    Returns
    -------
    dict
        State update with bias_analysis field

    Examples
    --------
    >>> state = BiasImageState(image_bytes=b"...", merged_text="...")
    >>> update = analyze_bias(state)
    >>> "bias_analysis" in update
    True
    """
    with telemetry.timer(
        "bias_image.analyze_bias",
        text_length=len(state.merged_text or ""),
    ):
        telemetry.log_info(
            "bias_image_analyze_start",
            text_length=len(state.merged_text or ""),
            run_id=state.run_id or "unknown",
        )

        try:
            # Get tools from registry
            registry = get_tool_registry()

            # Extract options
            merged_text = state.merged_text or ""
            temperature = state.options.get("temperature", 0.3)
            max_tokens = state.options.get("llm_max_tokens", 2000)

            # Validate parameters
            if not isinstance(temperature, (int, float)) or not (
                0.0 <= temperature <= 1.0
            ):
                telemetry.log_warning(
                    "invalid_temperature",
                    temperature=temperature,
                    using_default=0.3,
                )
                temperature = 0.3  # Use default for invalid values
            if not isinstance(max_tokens, int) or max_tokens <= 0:
                telemetry.log_warning(
                    "invalid_max_tokens",
                    max_tokens=max_tokens,
                    using_default=2000,
                )
                max_tokens = 2000  # Use default for invalid values

            # Phase 6.2: Load prompt template for bias analysis
            # Template instructs LLM to return precise character positions
            prompt_template_path = (
                Path(__file__).parent.parent
                / "prompts"
                / "templates"
                / "bias_analysis_v1.txt"
            )

            try:
                with open(prompt_template_path) as f:
                    prompt_template = f.read()
            except FileNotFoundError:
                telemetry.log_warning(
                    "prompt_template_not_found",
                    path=str(prompt_template_path),
                    using_fallback=True,
                )
                # Fallback to hardcoded prompt if template not found
                prompt_template = (
                    "You are a bias detection specialist analyzing text for "
                    "various forms of bias.\n\n"
                    "Your task is to carefully analyze the provided text and "
                    "identify ANY instances of bias across these categories:\n"
                    "- **Gender bias**: Gendered language, stereotypes, "
                    "exclusionary terms\n"
                    "- **Age bias**: Age discrimination, ageist assumptions, "
                    "generational stereotypes\n"
                    "- **Racial/ethnic bias**: Racial stereotypes, cultural "
                    "assumptions, exclusionary language\n"
                    "- **Disability bias**: Ableist language, accessibility "
                    "assumptions, capability stereotypes\n"
                    "- **Socioeconomic bias**: Class-based assumptions, "
                    "privilege bias, economic stereotypes\n\n"
                    "**TEXT TO ANALYZE:**\n"
                    "{text}\n\n"
                    "**IMPORTANT FOR IMAGE ANALYSIS:** Specify evidence_source as "
                    '"caption" or "ocr_text" for each bias instance.\n\n'
                    "Return valid JSON with structure: "
                    '{{"bias_detected": bool, "bias_instances": [...], '
                    '"overall_assessment": str, "risk_level": str}}'
                )

            # Build prompt with merged text
            prompt = prompt_template.format(text=merged_text)

            critique_feedback = state.options.get("bias_prompt_feedback")
            if isinstance(critique_feedback, list) and critique_feedback:
                feedback_section = "\n".join(f"- {item}" for item in critique_feedback)
                prompt += (
                    "\nEvaluator feedback to address:\n"
                    f"{feedback_section}\n"
                    "Ensure the refreshed analysis explicitly incorporates "
                    "the feedback above.\n"
                )

            # Call LLM via registry with structured output
            with telemetry.timer("bias_image.llm_call", temperature=temperature):
                llm_output = registry.llm.predict(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

            # Parse LLM output into BiasAnalysisOutput (same as bias_text_graph)
            if isinstance(llm_output, BiasAnalysisOutput):
                # Already structured (real LLM)
                bias_analysis = llm_output
                telemetry.log_info("bias_analysis_structured", format="pydantic_object")
            elif isinstance(llm_output, str):
                # JSON string (fake LLM) - parse into structured object
                try:
                    bias_analysis = BiasAnalysisOutput.model_validate_json(llm_output)
                    telemetry.log_info(
                        "bias_analysis_parsed",
                        format="json_to_pydantic",
                    )
                except Exception as e:
                    telemetry.log_error("bias_analysis_parse_failed", error=e)
                    raise ValueError(
                        f"Failed to parse LLM output into BiasAnalysisOutput: {e}",
                    ) from e
            else:
                raise TypeError(
                    f"Unexpected LLM output type: {type(llm_output)}. "
                    "Expected BiasAnalysisOutput or JSON string.",
                )

            telemetry.log_info(
                "bias_image_analyze_complete",
                bias_detected=bias_analysis.bias_detected,
                instance_count=len(bias_analysis.bias_instances),
            )
            return {"bias_analysis": bias_analysis}

        except Exception as e:
            telemetry.log_error("bias_image_analyze_bias_failed", error=e)
            raise


def summarize(state: BiasImageState) -> dict[str, str | None]:
    """Summarization node: Generate concise summary of bias findings.

    Creates a short executive summary of the bias analysis. Useful for
    quick insights or when merged text is long.

    Phase 6.4: Added telemetry for observability and performance tracking.

    Parameters
    ----------
    state : BiasImageState
        Current state with bias_analysis

    Returns
    -------
    dict
        State update with summary field

    Examples
    --------
    >>> state = BiasImageState(image_bytes=b"...", bias_analysis="...")
    >>> update = summarize(state)
    >>> "summary" in update
    True
    """
    with telemetry.timer("bias_image.summarize"):
        telemetry.log_info(
            "bias_image_summarize_start",
            run_id=state.run_id or "unknown",
        )

        try:
            # Get tools from registry
            registry = get_tool_registry()

            if state.bias_analysis is None:
                telemetry.log_warning("summarize_skipped", reason="no_bias_analysis")
                return {"summary": None}

            # Use summarizer tool to generate summary with error handling
            max_length = state.options.get("summary_max_length", 200)

            # Validate max_length parameter
            if not isinstance(max_length, int) or max_length <= 0:
                telemetry.log_warning(
                    "invalid_max_length",
                    max_length=max_length,
                    using_default=200,
                )
                max_length = 200  # Use default for invalid values

            try:
                # Convert BiasAnalysisOutput to string for summarization
                if isinstance(state.bias_analysis, BiasAnalysisOutput):
                    text_to_summarize = state.bias_analysis.overall_assessment
                else:
                    text_to_summarize = str(state.bias_analysis)

                with telemetry.timer("bias_image.summarizer_tool"):
                    summary = registry.summarizer.summarize(
                        text=text_to_summarize,
                        max_length=max_length,
                    )
                telemetry.log_info(
                    "bias_image_summarize_complete",
                    summary_length=len(summary) if summary else 0,
                )
            except Exception as e:
                telemetry.log_error("summarization_failed", error=e)
                # Fallback: None (summary is already optional in the workflow)
                # If summarization fails, workflow continues without summary
                summary = None

            return {"summary": summary}

        except Exception as e:
            telemetry.log_error("bias_image_summarize_failed", error=e)
            raise


def highlight(state: BiasImageState) -> dict[str, str]:
    """Highlighting node: Generate HTML with bias spans highlighted.

    Creates HTML representation of the merged text with problematic spans
    highlighted by bias type. Since this is image-derived text, the HTML
    includes both OCR and caption sections.

    Phase 6.2: Extracts precise character positions from bias_analysis
               with evidence source tracking (caption vs OCR).
    Phase 6.4: Added telemetry for observability and performance tracking.

    Parameters
    ----------
    state : BiasImageState
        Current state with merged_text and bias_analysis

    Returns
    -------
    dict
        State update with highlighted_html field

    Examples
    --------
    >>> state = BiasImageState(
    ...     image_bytes=b"...", merged_text="...", bias_analysis="..."
    ... )
    >>> update = highlight(state)
    >>> "<html>" in update["highlighted_html"]
    True
    """
    with telemetry.timer("bias_image.highlight"):
        telemetry.log_info(
            "bias_image_highlight_start",
            run_id=state.run_id or "unknown",
        )

        try:
            # Get tools from registry
            registry = get_tool_registry()

            merged_text = state.merged_text or ""

            # Phase 6.2: Extract spans from bias analysis with evidence source tracking
            spans: list[tuple[int, int, str]] = []

            if state.bias_analysis:
                # Extract (start_char, end_char, bias_type) tuples from analysis
                # This handles mapping from OCR/caption positions to merged_text
                with telemetry.timer("bias_image.extract_spans"):
                    spans = _extract_spans_from_analysis(
                        bias_analysis=state.bias_analysis,
                        merged_text=merged_text,
                        ocr_text=state.ocr_text or "",
                        caption_text=state.caption_text or "",
                    )
                telemetry.log_info("spans_extracted", span_count=len(spans))
            else:
                telemetry.log_warning(
                    "highlight_no_analysis",
                    reason="no_bias_analysis",
                )

            # Get bias type colors from configuration
            bias_types = settings.get_bias_type_colors()

            # Use formatter tool to generate highlighted HTML fragment
            # (dark-mode) with error handling
            try:
                with telemetry.timer("bias_image.formatter_tool"):
                    highlighted_html = registry.formatter.highlight_fragment(
                        text=merged_text,
                        spans=spans,
                        bias_types=bias_types,
                    )
                telemetry.log_info(
                    "bias_image_highlight_complete",
                    html_length=len(highlighted_html),
                )
            except Exception as e:
                telemetry.log_error("highlighting_failed", error=e)
                # Fallback: Plain HTML with unformatted text
                # Formatting is presentation layer - if it fails, provide plain text
                # rather than killing the entire workflow
                highlighted_html = f"<pre>{merged_text}</pre>"

            return {"highlighted_html": highlighted_html}

        except Exception as e:
            telemetry.log_error("bias_image_highlight_failed", error=e)
            raise


# Note: Quality assessment removed from subgraph
# Orchestrator's posthoc_eval node owns all evaluation (Phase 7+)


# ============================================================================
# Graph Construction
# ============================================================================


def create_bias_image_graph() -> CompiledStateGraph:
    """Create and compile the BiasImageGraph subgraph.

    Builds the image bias analysis workflow with parallel OCR and captioning.

    Graph Structure:
        START → extract_ocr (parallel with generate_caption)
        START → generate_caption (parallel with extract_ocr)
        Both → merge_text → analyze_bias → summarize → highlight → END

    Note: Quality assessment is handled by orchestrator's posthoc_eval node.

    Returns
    -------
    StateGraph
        Compiled bias image graph ready for invocation

    Examples
    --------
    >>> graph = create_bias_image_graph()
    >>> result = graph.invoke({"image_bytes": b"fake_image_data", "options": {}})
    >>> result["bias_analysis"]
    '...'
    """
    # Create graph with BiasImageState
    workflow = StateGraph(BiasImageState)

    # Add nodes
    workflow.add_node("extract_ocr", extract_ocr)
    workflow.add_node("generate_caption", generate_caption)
    workflow.add_node("merge_text", merge_text)
    workflow.add_node("analyze_bias", analyze_bias)
    workflow.add_node("summarize", summarize)
    workflow.add_node("highlight", highlight)

    # Add edges for parallel execution
    # START → both OCR and caption (parallel)
    workflow.add_edge(START, "extract_ocr")
    workflow.add_edge(START, "generate_caption")

    # Both parallel nodes → merge (fan-in)
    workflow.add_edge("extract_ocr", "merge_text")
    workflow.add_edge("generate_caption", "merge_text")

    # Sequential after merge
    workflow.add_edge("merge_text", "analyze_bias")
    workflow.add_edge("analyze_bias", "summarize")
    workflow.add_edge("summarize", "highlight")
    workflow.add_edge("highlight", END)

    # Compile graph
    return workflow.compile()
