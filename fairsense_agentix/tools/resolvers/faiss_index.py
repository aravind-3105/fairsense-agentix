"""Resolver for FAISS index tool."""

from pathlib import Path

from fairsense_agentix.tools.exceptions import ToolConfigurationError
from fairsense_agentix.tools.interfaces import EmbedderTool, FAISSIndexTool


def _resolve_faiss_tool(
    index_path: Path,
    embedder: EmbedderTool,
) -> FAISSIndexTool:
    """Resolve FAISS index tool.

    Phase 6.0: Updated to use LangChain FAISS vector store for better
    integration with retriever patterns and chain composition.

    Parameters
    ----------
    index_path : Path
        Path to FAISS index file (e.g., data/indexes/risks.faiss)
        For LangChain, this becomes a folder path (data/indexes/risks/)
    embedder : EmbedderTool
        Embedder tool for text-based search (must be LangChainEmbedder)

    Returns
    -------
    FAISSIndexTool
        Configured FAISS index tool instance (LangChainFAISSTool)

    Raises
    ------
    ToolConfigurationError
        If FAISS index cannot be loaded
    """
    # Check if using fake embedder - if so, always use fake FAISS
    # This handles testing scenarios where we have real indexes but fake tools
    from fairsense_agentix.tools.fake import FakeEmbedderTool  # noqa: PLC0415

    if isinstance(embedder, FakeEmbedderTool):
        from fairsense_agentix.tools.fake import FakeFAISSIndexTool  # noqa: PLC0415

        return FakeFAISSIndexTool(index_path=index_path, embedder=embedder)

    # Check if index exists - if not, use fake
    # For LangChain FAISS, check for folder instead of .faiss file
    index_folder = index_path.parent / index_path.stem  # data/indexes/risks
    langchain_index = index_folder / "index.faiss"  # LangChain format

    if not langchain_index.exists():
        from fairsense_agentix.tools.fake import FakeFAISSIndexTool  # noqa: PLC0415

        return FakeFAISSIndexTool(index_path=index_path, embedder=embedder)

    # Phase 6.0: Use LangChain FAISS vector store
    try:
        from fairsense_agentix.tools.embeddings import (  # noqa: PLC0415
            LangChainEmbedder,
        )
        from fairsense_agentix.tools.faiss_index import (  # noqa: PLC0415
            LangChainFAISSTool,
        )

        # Extract index name from path (e.g., "risks" from "risks.faiss")
        index_name = index_path.stem

        # LangChainFAISSTool requires LangChain embeddings, not just protocol
        if not isinstance(embedder, LangChainEmbedder):
            raise ToolConfigurationError(
                "LangChainFAISSTool requires LangChainEmbedder",
                context={
                    "embedder_type": type(embedder).__name__,
                    "expected": "LangChainEmbedder",
                },
            )

        return LangChainFAISSTool.load_local(
            folder_path=index_path.parent,  # data/indexes/
            index_name=index_name,  # "risks" or "rmf"
            embeddings=embedder.embeddings,  # Access underlying LangChain embeddings
        )
    except Exception as e:
        raise ToolConfigurationError(
            f"Failed to load LangChain FAISS index: {e}",
            context={
                "index_path": str(index_path),
                "index_folder": str(index_folder),
                "error_type": type(e).__name__,
            },
        ) from e
