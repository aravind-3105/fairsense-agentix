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

Node implementations and graph wiring live under
:mod:`fairsense_agentix.graphs.bias_image`.

Example:
    >>> from fairsense_agentix.graphs.bias_image_graph import create_bias_image_graph
    >>> graph = create_bias_image_graph()
    >>> result = graph.invoke({"image_bytes": b"fake_image_data", "options": {}})
    >>> result["bias_analysis"]
    '...'
"""

from fairsense_agentix.graphs.bias_image.build import create_bias_image_graph


__all__ = ["create_bias_image_graph"]
