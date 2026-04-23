"""Output schemas for Vision-Language Model (VLM) analysis.

This module defines structured output formats for VLM-based image bias detection,
including Chain-of-Thought reasoning traces for interpretability.
"""

from pydantic import BaseModel, Field

from fairsense_agentix.tools.llm.output_schemas import BiasAnalysisOutput


class BiasVisualAnalysisOutput(BaseModel):
    """VLM output with embedded Chain-of-Thought reasoning traces.

    Wraps the standard BiasAnalysisOutput with additional CoT metadata
    for interpretability and debugging. The CoT traces show the VLM's
    systematic reasoning process across 4 steps:
        1. Visual Inventory (Perceive)
        2. Pattern Analysis (Reason)
        3. Bias Detection (Evidence)
        4. Synthesis (Reflect)

    Attributes
    ----------
    visual_description : str
        Objective description of image contents (Step 1)
    reasoning_trace : str
        Analytical reasoning about patterns and relationships (Steps 2-3)
    bias_analysis : BiasAnalysisOutput
        Structured bias analysis with instances and risk assessment (Step 4)

    Examples
    --------
    >>> result = vlm.analyze_image(image_bytes, prompt, BiasVisualAnalysisOutput)
    >>> print(result.visual_description)
    'Image shows 3 people in a conference room...'
    >>> print(result.reasoning_trace)
    'Analyzing composition: 2 males centered, 1 female peripheral...'
    >>> print(result.bias_analysis.bias_detected)
    True
    >>> print(len(result.bias_analysis.bias_instances))
    2
    """

    visual_description: str = Field(
        description=(
            "Detailed objective description of image contents. "
            "Includes people, text, setting, objects, and compositional elements."
        ),
    )

    reasoning_trace: str = Field(
        description=(
            "Analytical reasoning about visual patterns and relationships. "
            "Documents the VLM's thought process for transparency."
        ),
    )

    bias_analysis: BiasAnalysisOutput = Field(
        description=(
            "Structured bias analysis output matching standard format. "
            "Includes bias instances, severity levels, and risk assessment."
        ),
    )
