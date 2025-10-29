"""BiasImageGraph subgraph for image bias analysis.

This workflow extracts information from images via parallel OCR and captioning,
merges the extracted text, then analyzes for bias. The parallel extraction
pattern enables faster processing.

Workflow:
    START → [extract_ocr ∥ generate_caption] → merge_text → analyze_bias
         → summarize → highlight → END

Phase 2 (Checkpoint 2.5): Stub implementations with fake OCR/caption/LLM
Phase 5+: Real OCR integration, captioning models, LLM analysis

Example:
    >>> from fairsense_agentix.graphs.bias_image_graph import create_bias_image_graph
    >>> graph = create_bias_image_graph()
    >>> result = graph.invoke({"image_bytes": b"fake_image_data", "options": {}})
    >>> result["bias_analysis"]
    '...'
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from fairsense_agentix.graphs.state import BiasImageState
from fairsense_agentix.tools import get_tool_registry


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
    # Get tools from registry
    registry = get_tool_registry()

    # Extract options
    language = state.options.get("ocr_language", "eng")
    confidence_threshold = state.options.get("ocr_confidence", 0.5)

    # Use OCR tool from registry with error handling
    try:
        ocr_text = registry.ocr.extract(
            image_bytes=state.image_bytes,
            language=language,
            confidence_threshold=confidence_threshold,
        )
    except Exception:
        # TODO Phase 8: Log OCR failure with telemetry
        # Fallback: Empty string allows workflow to continue with caption-only
        # OCR can fail for many reasons (bad image format, corrupted data, etc.)
        # Since caption runs in parallel, we can continue with single data source
        ocr_text = ""

    return {"ocr_text": ocr_text}


def generate_caption(state: BiasImageState) -> dict[str, str]:
    """Captioning node: Generate descriptive caption for image.

    Uses image captioning models (BLIP, BLIP-2, LLaVA, etc.) to generate
    a natural language description of the image content. This captures
    visual elements that OCR might miss.

    Executes in parallel with extract_ocr for faster processing.

    Phase 4+: Uses tool registry to get caption implementation (fake or real).
    Phase 5+: Real captioning using BLIP-2 or similar models.

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
    # Get tools from registry
    registry = get_tool_registry()

    # Extract options
    max_length = state.options.get("caption_max_length", 100)

    # Use caption tool from registry with error handling
    try:
        caption_text = registry.caption.caption(
            image_bytes=state.image_bytes, max_length=max_length
        )
    except Exception:
        # TODO Phase 8: Log caption failure with telemetry
        # Fallback: Empty string allows workflow to continue with OCR-only
        # Caption models can fail (model loading error, OOM, invalid image, etc.)
        # Since OCR runs in parallel, we can continue with single data source
        caption_text = ""

    return {"caption_text": caption_text}


def merge_text(state: BiasImageState) -> dict[str, str]:
    """Merge node: Combine OCR and caption text into unified input.

    Merges the extracted OCR text and generated caption into a single
    text input for bias analysis. Performs deduplication to avoid
    redundancy.

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
    # Phase 2: Simple concatenation
    # Phase 5+: Smart merging with deduplication and formatting

    ocr_text = state.ocr_text or ""
    caption_text = state.caption_text or ""

    # If BOTH extraction methods failed, we can't continue
    if not ocr_text and not caption_text:
        # TODO Phase 8: Log dual-extraction failure with telemetry
        # Raise error that orchestrator will catch and handle
        raise ValueError(
            "Both OCR and caption extraction failed - no text available for analysis"
        )

    # Simple merge with section headers
    merged = f"""**OCR Extracted Text:**
{ocr_text}

**Image Caption:**
{caption_text}"""

    return {"merged_text": merged}


def analyze_bias(state: BiasImageState) -> dict[str, str]:
    """LLM node: Analyze merged text for various forms of bias.

    Uses LLM to identify and explain bias in the merged OCR + caption text
    across multiple dimensions (gender, age, race, disability, socioeconomic).

    This is conceptually similar to BiasTextGraph's analyze_bias, but operates
    on the merged text from image analysis.

    Phase 4+: Uses tool registry to get LLM implementation (fake or real).
    Phase 5+: Real LLM call with bias detection prompt.

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
    # Get tools from registry
    registry = get_tool_registry()

    # Extract options
    merged_text = state.merged_text or ""
    temperature = state.options.get("temperature", 0.3)
    max_tokens = state.options.get("llm_max_tokens", 2000)

    # Build prompt for bias analysis (image-specific context)
    prompt = f"""Analyze the following text (extracted from an image via OCR and captioning) for various forms of bias including gender, age, racial/ethnic, disability, and socioeconomic bias.

Text to analyze:
{merged_text}

This text was extracted from an image, combining both visible text (OCR) and visual context (caption).

Provide a detailed analysis report with:
1. Bias Assessment (by category)
2. Visual Context Considerations
3. Overall Assessment
4. Recommendations for improvement
5. Confidence score

Format your response as a structured report."""

    # Call LLM via registry
    bias_analysis = registry.llm.predict(
        prompt=prompt, temperature=temperature, max_tokens=max_tokens
    )

    return {"bias_analysis": bias_analysis}


def summarize(state: BiasImageState) -> dict[str, str | None]:
    """Summarization node: Generate concise summary of bias findings.

    Creates a short executive summary of the bias analysis. Useful for
    quick insights or when merged text is long.

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
    # Get tools from registry
    registry = get_tool_registry()

    if state.bias_analysis is None:
        return {"summary": None}

    # Use summarizer tool to generate summary with error handling
    max_length = state.options.get("summary_max_length", 200)
    try:
        summary = registry.summarizer.summarize(
            text=state.bias_analysis, max_length=max_length
        )
    except Exception:
        # TODO Phase 8: Log summarization failure with telemetry
        # Fallback: None (summary is already optional in the workflow)
        # If summarization fails, workflow continues without summary
        summary = None

    return {"summary": summary}


def highlight(state: BiasImageState) -> dict[str, str]:
    """Highlighting node: Generate HTML with bias spans highlighted.

    Creates HTML representation of the merged text with problematic spans
    highlighted by bias type. Since this is image-derived text, the HTML
    includes both OCR and caption sections.

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
    # Get tools from registry
    registry = get_tool_registry()

    # Phase 4: Use formatter tool from registry
    # Phase 5+: Real highlighting logic will parse bias_analysis for spans

    merged_text = state.merged_text or ""

    # For now, pass empty spans list (Phase 5 will extract from bias_analysis)
    spans: list[tuple[int, int, str]] = []  # Future: extract from state.bias_analysis

    # Define bias type colors
    bias_types = {
        "gender": "#FFB3BA",
        "age": "#FFDFBA",
        "racial": "#FFFFBA",
        "disability": "#BAE1FF",
        "socioeconomic": "#E0BBE4",
    }

    # Use formatter tool to generate highlighted HTML with error handling
    try:
        highlighted_html = registry.formatter.highlight(
            text=merged_text, spans=spans, bias_types=bias_types
        )
    except Exception:
        # TODO Phase 8: Log formatting failure with telemetry
        # Fallback: Plain HTML with unformatted text
        # Formatting is presentation layer - if it fails, provide plain text
        # rather than killing the entire workflow
        highlighted_html = f"<pre>{merged_text}</pre>"

    return {"highlighted_html": highlighted_html}


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
