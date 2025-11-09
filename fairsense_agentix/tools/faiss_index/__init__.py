"""FAISS-based vector similarity search tools.

This module provides implementations for loading and searching FAISS indexes,
enabling fast semantic similarity search over large document collections.
"""

from fairsense_agentix.tools.faiss_index.faiss_index_tool import FAISSIndexTool


__all__ = ["FAISSIndexTool"]
