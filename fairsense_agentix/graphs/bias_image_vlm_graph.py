"""VLM-based image bias analysis graph.

Fast, intelligent image analysis using Vision-Language Models with embedded
Chain-of-Thought reasoning. Works with OpenAI GPT-4o Vision or Anthropic
Claude Sonnet Vision (determined by llm_provider setting).

This graph replaces the traditional OCR + Caption approach with direct VLM
analysis, providing:
    - 10-60x faster execution (2-5s vs 30-180s)
    - Systematic CoT reasoning traces
    - Higher quality bias detection
    - No GPU requirements (cloud API)
    - Unified provider model (same API key as text LLM)

Workflow:
    START → visual_analyze (VLM CoT) → summarize → highlight → END

Time: 2-5s vs 30-180s for traditional OCR+Caption approach

Implementation lives in :mod:`fairsense_agentix.graphs.bias_image_vlm`.
This module is the stable public entry point.
"""

from fairsense_agentix.graphs.bias_image_vlm.build import create_bias_image_vlm_graph


__all__ = ["create_bias_image_vlm_graph"]
