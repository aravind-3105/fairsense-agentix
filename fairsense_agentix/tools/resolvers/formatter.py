"""Resolver for formatter tool."""

from fairsense_agentix.configs.settings import Settings
from fairsense_agentix.tools.exceptions import ToolConfigurationError
from fairsense_agentix.tools.interfaces import FormatterTool


def _resolve_formatter_tool(settings: Settings) -> FormatterTool:
    """Resolve formatter tool.

    The formatter generates HTML with bias highlighting and data tables.
    It uses settings for bias color configuration.

    Parameters
    ----------
    settings : Settings
        Application configuration (includes bias color mappings)

    Returns
    -------
    FormatterTool
        Configured formatter tool instance

    Raises
    ------
    ToolConfigurationError
        If formatter cannot be constructed
    """
    # Phase 5.4: Use real HTML formatter
    try:
        from fairsense_agentix.tools.formatter import HTMLFormatter  # noqa: PLC0415

        # Get bias type colors from settings
        color_map = settings.get_bias_type_colors()

        return HTMLFormatter(color_map=color_map)

    except Exception as e:
        raise ToolConfigurationError(
            f"Failed to create HTMLFormatter: {e}",
            context={"error_type": type(e).__name__},
        ) from e
