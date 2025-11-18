"""LLM adapter implementations for FairSense-AgentiX (LangChain-based).

This package provides Large Language Model (LLM) adapters using LangChain:
- OpenAI (GPT-4, GPT-3.5-turbo) via langchain_openai.ChatOpenAI
- Anthropic (Claude 3.5 Sonnet, Claude 3 Opus) via langchain_anthropic.ChatAnthropic
- Local models (future: vLLM, llama.cpp)

All adapters implement the LLMTool protocol through LangChainLLMAdapter,
which wraps LangChain chat models with structured output parsing and telemetry.

Architecture:
- LangChainLLMAdapter: Protocol adapter wrapping LangChain models
- TelemetryCallback: Bridges LangChain callbacks → our telemetry service
- Output schemas: Pydantic models for validated LLM responses

Examples
--------
    >>> # Direct usage (typically done via registry)
    >>> from fairsense_agentix.tools.llm import LangChainLLMAdapter
    >>> from langchain_openai import ChatOpenAI
    >>> from langchain_core.output_parsers import PydanticOutputParser
    >>> from fairsense_agentix.tools.llm.output_schemas import BiasAnalysisOutput
    >>>
    >>> chat_model = ChatOpenAI(model="gpt-4", api_key="sk-...")
    >>> parser = PydanticOutputParser(pydantic_object=BiasAnalysisOutput)
    >>> adapter = LangChainLLMAdapter(
    ...     langchain_model=chat_model,
    ...     telemetry=telemetry,
    ...     settings=settings,
    ...     output_parser=parser,
    ... )
    >>> result = adapter.predict("Analyze for bias: ...")
    >>> assert isinstance(result, BiasAnalysisOutput)
"""

from fairsense_agentix.tools.llm.callbacks import TelemetryCallback
from fairsense_agentix.tools.llm.langchain_adapter import LangChainLLMAdapter
from fairsense_agentix.tools.llm.output_schemas import (
    BiasAnalysisOutput,
    BiasInstance,
    ImageBiasAnalysisOutput,
    RepresentationAnalysis,
)


__all__ = [
    "LangChainLLMAdapter",
    "TelemetryCallback",
    "BiasAnalysisOutput",
    "BiasInstance",
    "ImageBiasAnalysisOutput",
    "RepresentationAnalysis",
]
