"""BiasTextGraph subgraph for text bias analysis.

This workflow analyzes text input for various forms of bias (gender, age, race,
ability, etc.) using LLM-based analysis, optional summarization, and HTML
highlighting of problematic text spans.

Workflow:
    START → analyze_bias → summarize (conditional) → highlight → assess_quality → END

Phase 2 (Checkpoint 2.4): Stub implementations with fake LLM/tools
Phase 5+: Real LLM integration, prompt engineering, highlighting logic

Example:
    >>> from fairsense_agentix.graphs.bias_text_graph import create_bias_text_graph
    >>> graph = create_bias_text_graph()
    >>> result = graph.invoke(
    ...     {"text": "Sample job posting", "options": {"temperature": 0.3}}
    ... )
    >>> result["bias_analysis"]
    '...'
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from fairsense_agentix.graphs.state import BiasTextState


# ============================================================================
# Node Functions
# ============================================================================


def analyze_bias(state: BiasTextState) -> dict:
    """LLM node: Analyze text for various forms of bias.

    Uses LLM to identify and explain bias in text across multiple dimensions:
    - Gender bias
    - Age bias
    - Racial/ethnic bias
    - Disability bias
    - Socioeconomic bias

    In Phase 2 (Checkpoint 2.4), this is a stub that returns mock analysis.
    In Phase 5+, this will use real LLM calls with structured prompts.

    Parameters
    ----------
    state : BiasTextState
        Current state with text to analyze

    Returns
    -------
    dict
        State update with bias_analysis field

    Examples
    --------
    >>> state = BiasTextState(text="Job posting")
    >>> update = analyze_bias(state)
    >>> "bias_analysis" in update
    True
    """
    # Phase 2: Stub implementation with fake analysis
    # Phase 5+: Real LLM call with bias detection prompt

    text = state.text
    text_length = len(text)

    # Mock analysis based on text characteristics
    mock_analysis = f"""**Bias Analysis Report**

**Input Text**: {text[:100]}{"..." if text_length > 100 else ""}
**Text Length**: {text_length} characters

**Bias Assessment**:

1. **Gender Bias**: No explicit gender bias detected. The text uses neutral language.

2. **Age Bias**: No age-related bias detected. No age-specific terms or requirements found.

3. **Racial/Ethnic Bias**: No racial or ethnic bias detected. Language is inclusive.

4. **Disability Bias**: No disability-related bias detected.

5. **Socioeconomic Bias**: Minimal bias detected. Text is generally accessible.

**Overall Assessment**: The text demonstrates good awareness of potential bias.
Minor improvements could be made to enhance inclusivity.

**Recommendations**:
- Consider adding explicit diversity statements
- Review job requirements for hidden biases
- Ensure accessible language throughout

**Confidence**: 0.85
"""

    return {"bias_analysis": mock_analysis}


def summarize(state: BiasTextState) -> dict:
    """Summarization node: Generate concise summary of bias findings.

    Creates a short executive summary of the bias analysis, highlighting
    key findings and recommendations. This is useful for long texts or
    when users need quick insights.

    Conditional execution: Only runs if text is long (>500 chars) or if
    explicitly requested in options.

    Parameters
    ----------
    state : BiasTextState
        Current state with bias_analysis

    Returns
    -------
    dict
        State update with summary field

    Examples
    --------
    >>> state = BiasTextState(text="Long text...", bias_analysis="...")
    >>> update = summarize(state)
    >>> "summary" in update
    True
    """
    # Phase 2: Stub implementation
    # Phase 5+: Real summarization (LLM or extractive)

    if state.bias_analysis is None:
        return {"summary": None}

    # Mock summary
    mock_summary = """**Summary**: Minimal bias detected. Text is generally inclusive with
neutral language across gender, age, and racial dimensions. Consider adding explicit
diversity statements for improvement. Confidence: 0.85"""

    return {"summary": mock_summary}


def highlight(state: BiasTextState) -> dict:
    """Highlighting node: Generate HTML with bias spans highlighted.

    Creates an HTML representation of the input text with problematic
    spans highlighted in different colors based on bias type:
    - Gender bias: Pink
    - Age bias: Orange
    - Racial bias: Yellow
    - Disability bias: Blue
    - Socioeconomic bias: Purple

    Parameters
    ----------
    state : BiasTextState
        Current state with text and bias_analysis

    Returns
    -------
    dict
        State update with highlighted_html field

    Examples
    --------
    >>> state = BiasTextState(text="Sample text", bias_analysis="...")
    >>> update = highlight(state)
    >>> "<html>" in update["highlighted_html"]
    True
    """
    # Phase 2: Stub implementation with basic HTML
    # Phase 5+: Real highlighting logic based on bias analysis

    text = state.text

    # Mock HTML highlighting
    mock_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Bias Analysis - Highlighted Text</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 20px auto;
            padding: 20px;
            line-height: 1.6;
        }}
        .text-content {{
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 5px;
            border-left: 4px solid #4CAF50;
        }}
        .bias-gender {{ background-color: #FFB3BA; }}
        .bias-age {{ background-color: #FFDFBA; }}
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
    <h1>Bias Analysis - Highlighted Text</h1>

    <div class="text-content">
        <p>{text}</p>
    </div>

    <div class="legend">
        <h3>Bias Type Legend:</h3>
        <span class="legend-item bias-gender">Gender Bias</span>
        <span class="legend-item bias-age">Age Bias</span>
        <span class="legend-item bias-racial">Racial Bias</span>
        <span class="legend-item bias-disability">Disability Bias</span>
        <span class="legend-item bias-socioeconomic">Socioeconomic Bias</span>
    </div>

    <p><em>Note: This is a Phase 2 stub. Real highlighting will be implemented in Phase 5+.</em></p>
</body>
</html>"""

    return {"highlighted_html": mock_html}


# Note: Quality assessment removed from subgraph
# Orchestrator's posthoc_eval node owns all evaluation (Phase 7+)


# ============================================================================
# Conditional Edge Functions
# ============================================================================


def should_summarize(state: BiasTextState) -> bool:
    """Conditional edge: Decide whether to generate summary.

    Summarization is triggered if:
    - Text is long (>500 chars)
    - User explicitly requested it via options

    Parameters
    ----------
    state : BiasTextState
        Current state

    Returns
    -------
    bool
        True if summary should be generated
    """
    # Check text length
    if len(state.text) > 500:
        return True

    # Check options
    options = state.options or {}
    return bool(options.get("enable_summary", False))


# ============================================================================
# Graph Construction
# ============================================================================


def create_bias_text_graph() -> CompiledStateGraph:
    """Create and compile the BiasTextGraph subgraph.

    Builds the text bias analysis workflow with conditional summarization.

    Graph Structure:
        START → analyze_bias → summarize → highlight → END

    Note: Quality assessment is handled by orchestrator's posthoc_eval node.

    Returns
    -------
    StateGraph
        Compiled bias text graph ready for invocation

    Examples
    --------
    >>> graph = create_bias_text_graph()
    >>> result = graph.invoke({"text": "Sample job posting", "options": {}})
    >>> result["bias_analysis"]
    '...'
    """
    # Create graph with BiasTextState
    workflow = StateGraph(BiasTextState)

    # Add nodes
    workflow.add_node("analyze_bias", analyze_bias)
    workflow.add_node("summarize", summarize)
    workflow.add_node("highlight", highlight)

    # Add edges
    # START → analyze_bias
    workflow.add_edge(START, "analyze_bias")

    # analyze_bias → summarize
    workflow.add_edge("analyze_bias", "summarize")

    # summarize → highlight
    workflow.add_edge("summarize", "highlight")

    # highlight → END
    workflow.add_edge("highlight", END)

    # Compile graph
    return workflow.compile()
