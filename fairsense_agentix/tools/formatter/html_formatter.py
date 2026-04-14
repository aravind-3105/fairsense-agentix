"""HTML formatter facade for bias highlighting and data tables.

See :mod:`~fairsense_agentix.tools.formatter.highlight` for span highlighting
and :mod:`~fairsense_agentix.tools.formatter.tables` for tables.

:class:`HTMLFormatter` keeps the default color map and delegates to those modules.

Design Choices
--------------
1. **Why HTML escaping**: Prevents injection attacks from user-provided text
2. **Why complete HTML documents**: Self-contained, can be saved as standalone files
3. **Why inline CSS**: No external dependencies, guaranteed rendering
4. **Why span-based highlighting**: Semantic, accessible, easy to style

Examples
--------
    >>> formatter = HTMLFormatter(color_map={"gender": "#FFB3BA", "age": "#FFDFBA"})
    >>> html = formatter.highlight(
    ...     text="We need a young rockstar developer",
    ...     spans=[(11, 16, "age"), (17, 25, "gender")],
    ...     bias_types={"age": "#FFDFBA", "gender": "#FFB3BA"},
    ... )
    >>> with open("report.html", "w") as f:
    ...     f.write(html)
"""

from __future__ import annotations

import logging
from typing import Any

from fairsense_agentix.tools.exceptions import FormatterError

from . import highlight as highlight_html
from . import tables as tables_html


logger = logging.getLogger(__name__)


class HTMLFormatter:
    """HTML formatter for bias highlights and data tables.

    This formatter generates complete, self-contained HTML documents with inline
    CSS styling. It's designed to be safe (escapes HTML), accessible (semantic
    markup), and portable (no external dependencies).

    Parameters
    ----------
    color_map : dict[str, str] | None, optional
        Default color mapping for bias types (hex colors), by default None.
        If None, uses standard FairSense colors.

    Attributes
    ----------
    default_color_map : dict[str, str]
        Default colors for bias types (fallback if colors not provided)
    """

    def __init__(self, color_map: dict[str, str] | None = None) -> None:
        self.default_color_map = {
            "gender": "#FFB3BA",
            "age": "#FFDFBA",
            "racial": "#FFFFBA",
            "disability": "#BAE1FF",
            "socioeconomic": "#E0BBE4",
        }

        if color_map:
            self.default_color_map.update(color_map)

        logger.info(
            f"HTMLFormatter initialized with {len(self.default_color_map)} bias colors",
        )

    def highlight_fragment(
        self,
        text: str,
        spans: list[tuple[int, int, str]],
        bias_types: dict[str, str],
    ) -> str:
        """Generate dark-mode HTML fragment (not complete document) for embedding."""
        try:
            sorted_spans = sorted(spans, key=lambda s: (s[0], s[1]))
            html_output = highlight_html.build_dark_fragment(
                text,
                sorted_spans,
                bias_types,
            )

            logger.debug(
                f"Dark-mode HTML fragment generated "
                f"(text_length={len(text)}, spans={len(spans)})",
            )

            return html_output

        except Exception as e:
            msg = "Failed to generate dark-mode HTML fragment"
            raise FormatterError(
                msg,
                context={
                    "text_length": len(text) if isinstance(text, str) else None,
                    "span_count": len(spans) if isinstance(spans, list) else None,
                    "error": str(e),
                },
            ) from e

    def highlight(
        self,
        text: str,
        spans: list[tuple[int, int, str]],
        bias_types: dict[str, str],
    ) -> str:
        """Generate complete HTML document with highlighted text spans."""
        try:
            if not isinstance(text, str):
                raise FormatterError(
                    "Text must be a string",
                    context={"text_type": type(text).__name__},
                )

            sorted_spans = sorted(spans, key=lambda s: (s[0], s[1]))
            html_output = highlight_html.build_highlight_document(
                text,
                sorted_spans,
                bias_types,
            )

            logger.debug(
                f"Highlight HTML generated "
                f"(text_length={len(text)}, spans={len(spans)})",
            )

            return html_output

        except FormatterError:
            raise

        except Exception as e:
            msg = "Failed to generate highlight HTML"
            raise FormatterError(
                msg,
                context={
                    "text_length": len(text) if isinstance(text, str) else None,
                    "span_count": len(spans) if isinstance(spans, list) else None,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            ) from e

    def table(
        self,
        data: list[dict[str, Any]],
        headers: list[str] | None = None,
    ) -> str:
        """Generate complete HTML document containing a styled table."""
        try:
            if not data:
                logger.warning("Empty data provided to table formatter")
                return tables_html.build_empty_table_html()

            if headers is None:
                headers = list(data[0].keys())

            html_output = tables_html.build_table_html(data, headers)

            logger.debug(
                f"Table HTML generated (rows={len(data)}, columns={len(headers)})",
            )

            return html_output

        except FormatterError:
            raise

        except Exception as e:
            msg = "Failed to generate table HTML"
            raise FormatterError(
                msg,
                context={
                    "row_count": len(data) if isinstance(data, list) else None,
                    "header_count": len(headers) if headers else None,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            ) from e
