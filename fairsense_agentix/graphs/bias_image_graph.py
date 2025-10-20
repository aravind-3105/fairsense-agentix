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


# ============================================================================
# Node Functions
# ============================================================================


def extract_ocr(state: BiasImageState) -> dict:
    """OCR node: Extract text from image using OCR tool.

    Uses OCR (Optical Character Recognition) to extract text content from
    the image. Supports multiple OCR backends (Tesseract, PaddleOCR, etc.).

    Executes in parallel with generate_caption for faster processing.

    Phase 2: Stub returning mock OCR text.
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
    # Phase 2: Stub implementation
    # Phase 5+: Real OCR using Tesseract, PaddleOCR, or other backends

    image_size = len(state.image_bytes)

    # Mock OCR text based on image size
    if image_size > 10000:
        mock_ocr = """Job Posting: Senior Software Engineer

Requirements:
- 5+ years experience in software development
- Strong communication skills
- Bachelor's degree required
- Must be a recent graduate with fresh ideas

We're looking for energetic young professionals to join our team.
Candidates should be digital natives comfortable with modern technology."""
    else:
        mock_ocr = "Hiring Software Engineers\nApply today!"

    return {"ocr_text": mock_ocr}


def generate_caption(state: BiasImageState) -> dict:
    """Captioning node: Generate descriptive caption for image.

    Uses image captioning models (BLIP, BLIP-2, LLaVA, etc.) to generate
    a natural language description of the image content. This captures
    visual elements that OCR might miss.

    Executes in parallel with extract_ocr for faster processing.

    Phase 2: Stub returning mock caption.
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
    # Phase 2: Stub implementation
    # Phase 5+: Real captioning using BLIP-2, LLaVA, or other models

    image_size = len(state.image_bytes)

    # Mock caption based on image size
    if image_size > 10000:
        mock_caption = """A professional job posting advertisement featuring diverse team photos.
The image shows a modern office environment with collaborative workspaces.
Text overlays emphasize qualifications and company culture."""
    else:
        mock_caption = "A simple job posting flyer with text and company logo."

    return {"caption_text": mock_caption}


def merge_text(state: BiasImageState) -> dict:
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

    # Simple merge with section headers
    merged = f"""**OCR Extracted Text:**
{ocr_text}

**Image Caption:**
{caption_text}"""

    return {"merged_text": merged}


def analyze_bias(state: BiasImageState) -> dict:
    """LLM node: Analyze merged text for various forms of bias.

    Uses LLM to identify and explain bias in the merged OCR + caption text
    across multiple dimensions (gender, age, race, disability, socioeconomic).

    This is conceptually similar to BiasTextGraph's analyze_bias, but operates
    on the merged text from image analysis.

    Phase 2: Stub returning mock analysis.
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
    # Phase 2: Stub implementation
    # Phase 5+: Real LLM call with bias detection prompt

    merged_text = state.merged_text or ""
    text_length = len(merged_text)

    # Mock analysis
    mock_analysis = f"""**Bias Analysis Report (Image-Based)**

**Input Source**: Image analysis via OCR + Captioning
**Merged Text Length**: {text_length} characters

**Bias Assessment**:

1. **Gender Bias**: DETECTED - The phrase "digital natives" may inadvertently exclude
   older candidates. The emphasis on "recent graduate" combined with "fresh ideas" shows
   potential age discrimination.

2. **Age Bias**: HIGH - Multiple age-related red flags:
   - "Recent graduate" requirement is explicitly age-discriminatory
   - "Young professionals" directly references age
   - "Energetic" and "fresh" are common age-coded terms

3. **Racial/Ethnic Bias**: LOW - No explicit racial bias detected in the visible text.
   However, visual analysis notes diverse team photos, which is positive.

4. **Disability Bias**: LOW - No explicit disability-related bias detected.

5. **Socioeconomic Bias**: MODERATE - "Bachelor's degree required" may create barriers
   for qualified candidates from lower socioeconomic backgrounds.

**Visual Context** (from caption): The image portrays a modern, collaborative workplace.
While diverse representation is visible, the text content contradicts this inclusive imagery.

**Overall Assessment**: The text extracted from this image demonstrates concerning age bias.
The visual context suggests inclusivity, but the written content undermines this message.

**Recommendations**:
- Remove age-specific language ("young", "recent graduate")
- Replace "energetic" with skill-based requirements
- Consider making degree requirement flexible based on experience
- Align text content with inclusive visual messaging

**Confidence**: 0.88
"""

    return {"bias_analysis": mock_analysis}


def summarize(state: BiasImageState) -> dict:
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
    # Phase 2: Stub implementation
    # Phase 5+: Real summarization

    if state.bias_analysis is None:
        return {"summary": None}

    # Mock summary
    mock_summary = """**Summary**: HIGH age bias detected in image text. Job posting
explicitly targets "young professionals" and "recent graduates," which is discriminatory.
Visual context shows diversity, but written content contradicts this. Immediate revision
recommended to remove age-coded language. Confidence: 0.88"""

    return {"summary": mock_summary}


def highlight(state: BiasImageState) -> dict:
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
    # Phase 2: Stub implementation with basic HTML
    # Phase 5+: Real highlighting based on bias analysis

    merged_text = state.merged_text or ""

    # Mock HTML highlighting
    mock_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Bias Analysis - Image-Based Highlighting</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 900px;
            margin: 20px auto;
            padding: 20px;
            line-height: 1.6;
        }}
        .source-note {{
            background-color: #e3f2fd;
            padding: 15px;
            border-left: 4px solid #2196F3;
            margin-bottom: 20px;
            border-radius: 5px;
        }}
        .text-content {{
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 5px;
            border-left: 4px solid #4CAF50;
            white-space: pre-wrap;
        }}
        .bias-gender {{ background-color: #FFB3BA; }}
        .bias-age {{ background-color: #FFDFBA; font-weight: bold; }}
        .bias-racial {{ background-color: #FFFFBA; }}
        .bias-disability {{ background-color: #BAE1FF; }}
        .bias-socioeconomic {{ background-color: #E0BBE4; }}
        .legend {{
            margin-top: 20px;
            padding: 10px;
            background-color: #f0f0f0;
            border-radius: 5px;
        }}
        .legend-item {{
            display: inline-block;
            margin-right: 15px;
            padding: 5px 10px;
            border-radius: 3px;
        }}
    </style>
</head>
<body>
    <h1>Bias Analysis - Image-Based Text</h1>

    <div class="source-note">
        <strong>Source:</strong> Text extracted from image via OCR + Image Captioning<br>
        <strong>Note:</strong> Analysis combines both extracted text and visual context
    </div>

    <div class="text-content">
{merged_text}
    </div>

    <div class="legend">
        <h3>Bias Type Legend:</h3>
        <span class="legend-item bias-gender">Gender Bias</span>
        <span class="legend-item bias-age">Age Bias</span>
        <span class="legend-item bias-racial">Racial Bias</span>
        <span class="legend-item bias-disability">Disability Bias</span>
        <span class="legend-item bias-socioeconomic">Socioeconomic Bias</span>
    </div>

    <p><em>Note: Phase 2 stub. Real highlighting will identify specific problematic spans in Phase 5+.</em></p>
</body>
</html>"""

    return {"highlighted_html": mock_html}


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
