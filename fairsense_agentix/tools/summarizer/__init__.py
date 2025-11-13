"""Summarizer tools for FairSense-AgentiX.

This module provides text summarization tools for condensing lengthy analysis
outputs into concise, actionable summaries.

Examples
--------
    >>> from fairsense_agentix.tools.summarizer import LLMSummarizer
    >>> from langchain_openai import ChatOpenAI
    >>> from fairsense_agentix.services.telemetry import TelemetryService
    >>> from fairsense_agentix.configs import settings
    >>> from fairsense_agentix.prompts import PromptLoader
    >>>
    >>> model = ChatOpenAI(model="gpt-4")
    >>> telemetry = TelemetryService(enabled=True)
    >>> summarizer = LLMSummarizer(
    ...     langchain_model=model,
    ...     telemetry=telemetry,
    ...     settings=settings,
    ...     prompt_loader=PromptLoader(),
    ... )
    >>> summary = summarizer.summarize(long_text, max_length=200)
"""

from fairsense_agentix.tools.summarizer.llm_summarizer import LLMSummarizer


__all__ = ["LLMSummarizer"]
