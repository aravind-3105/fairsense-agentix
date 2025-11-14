"""Embedding tools for semantic text representation.

This module provides implementations of the EmbedderTool protocol for
converting text into dense vector representations suitable for semantic
similarity search.
"""

from fairsense_agentix.tools.embeddings.langchain_embedder import LangChainEmbedder
from fairsense_agentix.tools.embeddings.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
)


__all__ = [
    "SentenceTransformerEmbedder",
    "LangChainEmbedder",
]
