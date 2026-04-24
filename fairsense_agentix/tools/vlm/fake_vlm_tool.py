"""Fake Vision-Language Model tool for testing without API calls.

This module provides a mock VLM implementation that returns deterministic
outputs without making actual API calls. Useful for:
    - Unit testing VLM-based workflows
    - Development without API keys
    - CI/CD pipelines without external dependencies
"""

import logging

from pydantic import BaseModel


logger = logging.getLogger(__name__)


class FakeVLMTool:
    """Fake VLM that returns mock structured output.

    Implements the VLMTool protocol without actual API calls. Returns
    realistic mock data for testing VLM-based bias detection workflows.

    Examples
    --------
    >>> vlm = FakeVLMTool()
    >>> result = vlm.analyze_image(
    ...     image_bytes=b"fake_image",
    ...     prompt="Analyze for bias",
    ...     response_model=BiasVisualAnalysisOutput,
    ... )
    >>> assert isinstance(result, BiasVisualAnalysisOutput)
    >>> assert result.bias_analysis.bias_detected is True
    """

    def analyze_image(
        self,
        image_bytes: bytes,
        prompt: str,
        response_model: type[BaseModel],
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> BaseModel:
        """Return mock BiasVisualAnalysisOutput without API call.

        Parameters
        ----------
        image_bytes : bytes
            Image data (not actually processed)
        prompt : str
            Analysis prompt (not actually used)
        response_model : type[BaseModel]
            Expected output schema (should be BiasVisualAnalysisOutput)
        temperature : float, optional
            Not used (fake tool is deterministic)
        max_tokens : int, optional
            Not used (fake tool has fixed output)

        Returns
        -------
        BaseModel
            Mock BiasVisualAnalysisOutput instance

        Notes
        -----
        Always returns the same mock output with 2 bias instances
        (1 gender, 1 age) for predictable testing.
        """
        logger.info(f"FakeVLM: Analyzing {len(image_bytes)} byte image (mock analysis)")

        # Import here to avoid circular dependency
        from fairsense_agentix.tools.llm.output_schemas import (
            BiasAnalysisOutput,
            BiasInstance,
        )
        from fairsense_agentix.tools.vlm.output_schemas import (
            BiasVisualAnalysisOutput,
        )

        # Create deterministic fake output
        fake_output = BiasVisualAnalysisOutput(
            visual_description=(
                "Image shows 3 people in a modern office environment. "
                "Two individuals appear male-presenting (ages ~35-40), "
                "positioned centrally and standing at a whiteboard. "
                "One individual appears female-presenting (age ~30), seated at a "
                "desk in the background. Professional attire visible on all "
                "individuals. Whiteboard contains text 'Q4 Strategy Meeting'. "
                "Setting suggests corporate context."
            ),
            reasoning_trace=(
                "Analyzing compositional patterns: The two male-presenting "
                "individuals occupy central, prominent positions and are engaged "
                "in active presentation behavior. The female-presenting individual "
                "is spatially peripheral and in a passive, note-taking posture. "
                "This spatial arrangement reinforces traditional gender role "
                "stereotypes (men as leaders/speakers, women as support/listeners). "
                "Age representation is narrow (30-40 range), suggesting potential "
                "age bias through exclusion of younger or older individuals in "
                "leadership context."
            ),
            bias_analysis=BiasAnalysisOutput(
                bias_detected=True,
                bias_instances=[
                    BiasInstance(
                        type="gender",
                        severity="medium",
                        text_span=(
                            "Male-dominated leadership representation with women in "
                            "peripheral support role"
                        ),
                        explanation=(
                            "Visual composition shows 2 male-presenting individuals "
                            "in central authoritative positions (standing, presenting "
                            "at whiteboard) while 1 female-presenting individual is "
                            "peripheral and in passive role (seated, note-taking). "
                            "This reinforces gender stereotypes about leadership and "
                            "professional roles."
                        ),
                        start_char=0,
                        end_char=0,
                        evidence_source="visual",
                    ),
                    BiasInstance(
                        type="age",
                        severity="low",
                        text_span=(
                            "Narrow age representation (30-40 range) in leadership "
                            "context"
                        ),
                        explanation=(
                            "All visible individuals appear to be in the 30-40 age "
                            "range, suggesting exclusion of both younger professionals "
                            "(potential ageism about credibility/experience) and older "
                            "individuals (potential ageism about relevance/"
                            "adaptability) from leadership representation."
                        ),
                        start_char=0,
                        end_char=0,
                        evidence_source="visual",
                    ),
                ],
                overall_assessment=(
                    "Image exhibits moderate bias through gender-stereotyped role "
                    "representation and narrow age range in professional leadership "
                    "context. The 2:1 male-to-female ratio combined with "
                    "spatial/postural hierarchy reinforces traditional gender norms. "
                    "Age homogeneity suggests exclusionary assumptions about "
                    "leadership demographics."
                ),
                risk_level="medium",
            ),
        )

        logger.info(
            "FakeVLM: Returned mock analysis with 2 bias instances (1 gender, 1 age)",
        )

        return fake_output
