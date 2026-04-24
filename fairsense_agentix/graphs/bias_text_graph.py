"""BiasTextGraph subgraph for text bias analysis.

This workflow analyzes text input for various forms of bias (gender, age, race,
ability, etc.) using LLM-based analysis, optional summarization, and HTML
highlighting of problematic text spans.

Workflow:
    START → analyze_bias → [summarize (conditional)] → highlight → END

    Conditional: summarize runs if text > 500 chars or options["enable_summary"] = True

Note: Quality assessment is handled by orchestrator's posthoc_eval node.
Subgraphs produce output; orchestrator evaluates quality.

Phase 2 (Checkpoint 2.4): Stub implementations with fake LLM/tools
Phase 5+: Real LLM integration, prompt engineering, highlighting logic
Phase 6.1: Span extraction with character-level precision

Implementation lives under :mod:`fairsense_agentix.graphs.bias_text`.

Example:
    >>> from fairsense_agentix.graphs.bias_text_graph import create_bias_text_graph
    >>> graph = create_bias_text_graph()
    >>> result = graph.invoke(
    ...     {"text": "Sample job posting", "options": {"temperature": 0.3}}
    ... )
    >>> result["bias_analysis"]
    '...'
"""

from fairsense_agentix.graphs.bias_text.build import create_bias_text_graph


__all__ = ["create_bias_text_graph"]
