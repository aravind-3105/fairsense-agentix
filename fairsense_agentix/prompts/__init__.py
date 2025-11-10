"""Prompt management module for FairSense-AgentiX.

This module provides infrastructure for loading and rendering prompt templates.
Prompts are stored as versioned text files with variable substitution support.

Examples
--------
    >>> from fairsense_agentix.prompts import PromptLoader
    >>> loader = PromptLoader()
    >>> prompt = loader.load("bias_text_v1", text="Sample text to analyze")
    >>> print(prompt)
    You are an AI bias detection expert. Analyze the following text...
"""

from fairsense_agentix.prompts.prompt_loader import PromptLoader


__all__ = ["PromptLoader"]
