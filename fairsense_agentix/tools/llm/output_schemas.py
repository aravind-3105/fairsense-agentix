"""Pydantic schemas for structured LLM outputs.

This module defines the expected output structures for LLM responses.
These schemas are used with LangChain's PydanticOutputParser to validate
and parse LLM outputs at runtime.

Benefits:
- Type safety: LLM outputs are validated Python objects, not raw strings
- Runtime validation: Invalid responses fail immediately with clear errors
- Self-documenting: Schemas define the contract between LLM and application
- IDE support: Autocomplete and type checking work seamlessly

Examples
--------
    >>> from langchain_core.output_parsers import PydanticOutputParser
    >>> parser = PydanticOutputParser(pydantic_object=BiasAnalysisOutput)
    >>> # Parser automatically adds format instructions to prompts
    >>> format_instructions = parser.get_format_instructions()
    >>> # Parse LLM response into validated object
    >>> result = parser.parse(llm_output)
    >>> assert isinstance(result, BiasAnalysisOutput)
"""

from typing import Literal

from pydantic import BaseModel, Field


class BiasInstance(BaseModel):
    """Single instance of detected bias in text or image.

    Attributes
    ----------
    type : str
        Bias category: "gender", "age", "racial", "disability", or "socioeconomic"
    severity : str
        Impact severity: "low" (subtle/implicit), "medium" (clear), or "high" (overt)
    text_span : str
        Exact quote from text showing the bias (for text analysis)
    visual_element : str
        Description of visual element showing bias (for image analysis)
    explanation : str
        Why this constitutes bias (evidence-based reasoning)
    start_char : int
        Start position in text (0-indexed), for text highlighting
    end_char : int
        End position in text (0-indexed), for text highlighting
    evidence_source : str
        Source of evidence: "text", "caption", or "ocr_text"

    Examples
    --------
    >>> instance = BiasInstance(
    ...     type="gender",
    ...     severity="high",
    ...     text_span="salesman needed",
    ...     explanation="Uses gendered term instead of gender-neutral 'salesperson'",
    ...     start_char=0,
    ...     end_char=16,
    ... )
    """

    type: Literal["gender", "age", "racial", "disability", "socioeconomic"] = Field(
        ...,
        description="Bias category: gender, age, racial, disability, or socioeconomic",
    )

    severity: Literal["low", "medium", "high"] = Field(
        ...,
        description="Impact severity: low (subtle), medium (clear), high (overt)",
    )

    text_span: str = Field(
        default="",
        description="Exact quote from text showing bias (empty for image-only bias)",
    )

    visual_element: str = Field(
        default="",
        description="Description of visual element showing bias (for images)",
    )

    explanation: str = Field(
        ...,
        description="Why this constitutes bias with evidence-based reasoning",
    )

    start_char: int = Field(
        default=0,
        ge=0,
        description="Start position in text (0-indexed) for highlighting",
    )

    end_char: int = Field(
        default=0,
        ge=0,
        description="End position in text (0-indexed) for highlighting",
    )

    evidence_source: Literal["text", "caption", "ocr_text", "visual"] = Field(
        default="text",
        description=(
            "Source of evidence: "
            "text (direct text analysis), "
            "caption (image caption), "
            "ocr_text (OCR extraction), "
            "visual (direct VLM visual analysis)"
        ),
    )


class BiasAnalysisOutput(BaseModel):
    """Structured output for bias analysis from LLM.

    This schema is used for both text and image bias analysis. The LLM must
    return JSON matching this structure, validated at runtime by Pydantic.

    Attributes
    ----------
    bias_detected : bool
        Whether any bias was found
    bias_instances : list[BiasInstance]
        List of specific bias instances found (empty if none)
    overall_assessment : str
        Brief summary of findings (1-3 sentences)
    risk_level : str
        Overall risk: "low", "medium", or "high"

    Examples
    --------
    >>> output = BiasAnalysisOutput(
    ...     bias_detected=True,
    ...     bias_instances=[
    ...         BiasInstance(
    ...             type="gender",
    ...             severity="high",
    ...             text_span="salesman",
    ...             explanation="Gendered term excludes women",
    ...             start_char=0,
    ...             end_char=8,
    ...         )
    ...     ],
    ...     overall_assessment="Job posting contains gender bias in role title",
    ...     risk_level="high",
    ... )
    """

    bias_detected: bool = Field(
        ...,
        description="Whether any bias was detected in the content",
    )

    bias_instances: list[BiasInstance] = Field(
        default_factory=list,
        description="List of specific bias instances (empty if none found)",
    )

    overall_assessment: str = Field(
        ...,
        description="Brief summary of bias findings (1-3 sentences)",
    )

    risk_level: Literal["low", "medium", "high"] = Field(
        ...,
        description="Overall risk level: low, medium, or high",
    )


class RepresentationAnalysis(BaseModel):
    """Representation analysis for image bias detection.

    Analyzes diversity and inclusion in visual content.

    Attributes
    ----------
    diversity_score : str
        Overall diversity: "low", "medium", or "high"
    underrepresented_groups : list[str]
        Groups with insufficient representation
    overrepresented_groups : list[str]
        Groups with dominant representation

    Examples
    --------
    >>> analysis = RepresentationAnalysis(
    ...     diversity_score="low",
    ...     underrepresented_groups=["women", "people with disabilities"],
    ...     overrepresented_groups=["young white men"],
    ... )
    """

    diversity_score: Literal["low", "medium", "high"] = Field(
        ...,
        description="Overall diversity in visual content",
    )

    underrepresented_groups: list[str] = Field(
        default_factory=list,
        description="Groups with insufficient representation",
    )

    overrepresented_groups: list[str] = Field(
        default_factory=list,
        description="Groups with dominant representation",
    )


class ImageBiasAnalysisOutput(BaseModel):
    """Structured output for image bias analysis from LLM.

    Extends base bias analysis with representation metrics specific to images.

    Attributes
    ----------
    bias_detected : bool
        Whether any bias was found
    bias_instances : list[BiasInstance]
        List of specific bias instances found (empty if none)
    representation_analysis : RepresentationAnalysis
        Diversity and representation metrics
    overall_assessment : str
        Brief summary of findings (1-3 sentences)
    risk_level : str
        Overall risk: "low", "medium", or "high"

    Examples
    --------
    >>> output = ImageBiasAnalysisOutput(
    ...     bias_detected=True,
    ...     bias_instances=[...],
    ...     representation_analysis=RepresentationAnalysis(
    ...         diversity_score="low",
    ...         underrepresented_groups=["women"],
    ...         overrepresented_groups=["men"],
    ...     ),
    ...     overall_assessment="Image shows gender imbalance in leadership roles",
    ...     risk_level="medium",
    ... )
    """

    bias_detected: bool = Field(
        ...,
        description="Whether any bias was detected in the image",
    )

    bias_instances: list[BiasInstance] = Field(
        default_factory=list,
        description="List of specific bias instances (empty if none found)",
    )

    representation_analysis: RepresentationAnalysis = Field(
        ...,
        description="Diversity and representation analysis",
    )

    overall_assessment: str = Field(
        ...,
        description="Brief summary of visual bias findings (1-3 sentences)",
    )

    risk_level: Literal["low", "medium", "high"] = Field(
        ...,
        description="Overall risk level: low, medium, or high",
    )
