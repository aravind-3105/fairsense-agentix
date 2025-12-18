"""Vision-Language Model (VLM) tools for image analysis.

This module provides VLM implementations that analyze images with structured
output and Chain-of-Thought reasoning. Supports both OpenAI (GPT-4o Vision)
and Anthropic (Claude Sonnet Vision) providers.

Key Features:
    - Unified provider model (uses llm_provider setting)
    - Structured output via Pydantic models
    - Chain-of-Thought reasoning traces
    - Fake implementation for testing without API calls

Examples
--------
    >>> from fairsense_agentix.tools.vlm import UnifiedVLMTool
    >>> from fairsense_agentix.configs import settings
    >>>
    >>> vlm = UnifiedVLMTool(settings)
    >>> result = vlm.analyze_image(
    ...     image_bytes=image_data,
    ...     prompt="Analyze for bias",
    ...     response_model=BiasVisualAnalysisOutput,
    ... )
    >>> print(result.bias_analysis.overall_assessment)
"""

from fairsense_agentix.tools.vlm.fake_vlm_tool import FakeVLMTool
from fairsense_agentix.tools.vlm.output_schemas import BiasVisualAnalysisOutput
from fairsense_agentix.tools.vlm.unified_vlm_tool import UnifiedVLMTool


__all__ = [
    "UnifiedVLMTool",
    "FakeVLMTool",
    "BiasVisualAnalysisOutput",
]
