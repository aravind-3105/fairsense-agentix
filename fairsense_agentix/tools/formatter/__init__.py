"""Formatter tools for FairSense-AgentiX.

This module provides HTML formatting tools for generating styled output including
highlighted text with bias spans and structured data tables.

Examples
--------
    >>> from fairsense_agentix.tools.formatter import HTMLFormatter
    >>>
    >>> formatter = HTMLFormatter(color_map={"gender": "#FFB3BA", "age": "#FFDFBA"})
    >>>
    >>> # Highlight bias spans
    >>> html = formatter.highlight(
    ...     text="Looking for a young rockstar developer",
    ...     spans=[(15, 20, "age")],  # "young"
    ...     bias_types={"age": "#FFDFBA"},
    ... )
    >>>
    >>> # Generate data table
    >>> table_html = formatter.table(
    ...     data=[{"risk": "Bias", "severity": "High"}],
    ...     headers=["Risk", "Severity"],
    ... )
"""

from fairsense_agentix.tools.formatter.html_formatter import HTMLFormatter


__all__ = ["HTMLFormatter"]
