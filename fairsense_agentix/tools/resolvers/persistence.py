"""Resolver for persistence tool."""

from pathlib import Path

from fairsense_agentix.tools.exceptions import ToolConfigurationError
from fairsense_agentix.tools.interfaces import PersistenceTool


def _resolve_persistence_tool(output_dir: Path) -> PersistenceTool:
    """Resolve persistence tool.

    Parameters
    ----------
    output_dir : Path
        Directory for saved files

    Returns
    -------
    PersistenceTool
        Configured persistence tool instance

    Raises
    ------
    ToolConfigurationError
        If persistence tool cannot be constructed
    """
    # Phase 5.4: Use real CSV writer
    try:
        from fairsense_agentix.tools.persistence import CSVWriter  # noqa: PLC0415

        return CSVWriter(output_dir=output_dir)

    except Exception as e:
        raise ToolConfigurationError(
            f"Failed to create CSVWriter: {e}",
            context={
                "output_dir": str(output_dir),
                "error_type": type(e).__name__,
            },
        ) from e
