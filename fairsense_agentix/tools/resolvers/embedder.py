"""Resolver for embedder tool."""

from fairsense_agentix.configs.settings import Settings
from fairsense_agentix.tools.exceptions import ToolConfigurationError
from fairsense_agentix.tools.interfaces import EmbedderTool


def _resolve_embedder_tool(settings: Settings) -> EmbedderTool:
    """Resolve embedder tool from settings.

    Phase 6.0: Updated to use LangChain HuggingFaceEmbeddings for better
    integration with LangChain FAISS and retriever patterns.

    Beta Release: Added telemetry support for model download visibility.
    Uses global telemetry service to emit download progress events during
    server startup (when embedder is first initialized). This provides
    transparency for the 30-120s model download on first use.

    Parameters
    ----------
    settings : Settings
        Application configuration

    Returns
    -------
    EmbedderTool
        Configured embedder tool instance (LangChainEmbedder)

    Raises
    ------
    ToolConfigurationError
        If embedder cannot be constructed
    """
    # Check if using fake embedder
    if settings.embedding_model == "fake":
        from fairsense_agentix.tools.fake import FakeEmbedderTool  # noqa: PLC0415

        return FakeEmbedderTool(dimension=settings.embedding_dimension)

    # Phase 6.0: Use LangChain HuggingFaceEmbeddings wrapper
    # Beta: Add telemetry support for model download progress
    try:
        # Import global telemetry service for server-level events
        from fairsense_agentix.services.telemetry import (  # noqa: PLC0415
            TelemetryService,
        )
        from fairsense_agentix.tools.embeddings import (  # noqa: PLC0415
            LangChainEmbedder,
        )

        # Create telemetry service for model loading events
        # Uses server-level logging (not per-request, since registry is singleton)
        telemetry = TelemetryService(enabled=settings.telemetry_enabled)

        # Generate server startup ID for tracing model loads
        import uuid  # noqa: PLC0415

        server_startup_id = f"server_startup_{uuid.uuid4().hex[:8]}"

        return LangChainEmbedder(
            model_name=settings.embedding_model,
            dimension=settings.embedding_dimension,
            normalize=True,  # For cosine similarity (FAISS IndexFlatIP)
            telemetry=telemetry,  # Enable model download progress events
            run_id=server_startup_id,  # Server-level trace ID
        )
    except Exception as e:
        raise ToolConfigurationError(
            f"Failed to create LangChainEmbedder: {e}",
            context={
                "model_name": settings.embedding_model,
                "dimension": settings.embedding_dimension,
                "error_type": type(e).__name__,
            },
        ) from e
