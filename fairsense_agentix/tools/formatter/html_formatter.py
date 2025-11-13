"""HTML formatter implementation for bias highlighting and data tables.

This module provides production-ready HTML generation for visualizing bias detection
results and structured data. It includes proper HTML escaping, CSS styling, and
handles edge cases like overlapping spans and empty data.

Design Choices
--------------
1. **Why HTML escaping**: Prevents injection attacks from user-provided text
2. **Why complete HTML documents**: Self-contained, can be saved as standalone files
3. **Why inline CSS**: No external dependencies, guaranteed rendering
4. **Why span-based highlighting**: Semantic, accessible, easy to style

What This Enables
-----------------
- Safe HTML generation from untrusted text inputs
- Portable HTML outputs (can be opened directly in browsers)
- Visual bias analysis reports for non-technical stakeholders
- Structured risk assessment tables for documentation

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

import html
import logging
from typing import Any

from fairsense_agentix.tools.exceptions import FormatterError


logger = logging.getLogger(__name__)


class HTMLFormatter:
    """HTML formatter for bias highlights and data tables.

    This formatter generates complete, self-contained HTML documents with inline
    CSS styling. It's designed to be safe (escapes HTML), accessible (semantic
    markup), and portable (no external dependencies).

    Design Philosophy:
    - **Safety first**: Always escape user text to prevent HTML injection
    - **Self-contained**: Complete HTML documents with inline styles
    - **Accessibility**: Semantic HTML with proper structure
    - **Edge case handling**: Empty data, overlapping spans, invalid inputs

    Parameters
    ----------
    color_map : dict[str, str] | None, optional
        Default color mapping for bias types (hex colors), by default None.
        If None, uses standard FairSense colors.

    Attributes
    ----------
    default_color_map : dict[str, str]
        Default colors for bias types (fallback if colors not provided)

    Examples
    --------
    >>> # With custom colors
    >>> formatter = HTMLFormatter(color_map={"gender": "#FF0000"})
    >>>
    >>> # With default colors
    >>> formatter = HTMLFormatter()
    >>>
    >>> # Highlight bias spans
    >>> html = formatter.highlight(
    ...     text="Looking for a young developer",
    ...     spans=[(15, 20, "age")],
    ...     bias_types={"age": "#FFDFBA"},
    ... )

    Notes
    -----
    The formatter always returns complete HTML documents (DOCTYPE, html, head, body).
    This makes outputs self-contained and directly openable in browsers.
    """

    def __init__(self, color_map: dict[str, str] | None = None) -> None:
        """Initialize HTML formatter.

        Parameters
        ----------
        color_map : dict[str, str] | None, optional
            Default color mapping for bias types, by default None
        """
        # Default FairSense colors (fallback)
        self.default_color_map = {
            "gender": "#FFB3BA",
            "age": "#FFDFBA",
            "racial": "#FFFFBA",
            "disability": "#BAE1FF",
            "socioeconomic": "#E0BBE4",
        }

        # Override defaults with provided colors
        if color_map:
            self.default_color_map.update(color_map)

        logger.info(
            f"HTMLFormatter initialized with {len(self.default_color_map)} bias colors"
        )

    def highlight(
        self,
        text: str,
        spans: list[tuple[int, int, str]],
        bias_types: dict[str, str],
    ) -> str:
        """Generate HTML with highlighted text spans.

        Creates a complete HTML document with the input text and bias spans
        highlighted using color-coded backgrounds. Text is properly HTML-escaped
        to prevent injection attacks.

        Parameters
        ----------
        text : str
            Text to highlight (will be HTML-escaped for safety)
        spans : list[tuple[int, int, str]]
            List of (start_idx, end_idx, bias_type) tuples defining spans to highlight.
            Indices are 0-based character positions in the original text.
        bias_types : dict[str, str]
            Mapping of bias_type to hex color code (e.g., {"gender": "#FFB3BA"})

        Returns
        -------
        str
            Complete HTML document with highlighted spans

        Raises
        ------
        FormatterError
            If formatting fails (invalid spans, template error, etc.)

        Examples
        --------
        >>> formatter = HTMLFormatter()
        >>> html = formatter.highlight(
        ...     text="Looking for a rockstar developer",
        ...     spans=[(15, 23, "gender")],  # "rockstar"
        ...     bias_types={"gender": "#FFB3BA"},
        ... )
        >>> "rockstar" in html
        True

        Notes
        -----
        - Overlapping spans are handled by sorting and applying in order
        - Invalid span indices (out of bounds) are logged and skipped
        - Text is HTML-escaped before processing to prevent injection
        """
        try:
            # Validate inputs
            if not isinstance(text, str):
                raise FormatterError(
                    "Text must be a string",
                    context={"text_type": type(text).__name__},
                )

            # Sort spans by start position (handle overlaps consistently)
            sorted_spans = sorted(spans, key=lambda s: (s[0], s[1]))

            # Build HTML with proper escaping
            html_output = self._build_highlight_html(text, sorted_spans, bias_types)

            logger.debug(
                f"Highlight HTML generated (text_length={len(text)}, spans={len(spans)})"
            )

            return html_output

        except FormatterError:
            # Re-raise our own errors unchanged
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
        """Generate HTML table from structured data.

        Creates a complete HTML document containing a styled table with the
        provided data. Headers are inferred from the first row if not specified.

        Parameters
        ----------
        data : list[dict[str, Any]]
            List of row dictionaries (keys become columns)
        headers : list[str] | None, optional
            Column headers (if None, infer from first row keys), by default None

        Returns
        -------
        str
            Complete HTML document with styled table

        Raises
        ------
        FormatterError
            If formatting fails (empty data, invalid structure, etc.)

        Examples
        --------
        >>> formatter = HTMLFormatter()
        >>> html = formatter.table(
        ...     data=[
        ...         {"risk": "Bias", "severity": "High", "score": 0.89},
        ...         {"risk": "Privacy", "severity": "Medium", "score": 0.65},
        ...     ],
        ...     headers=["Risk", "Severity", "Score"],
        ... )
        >>> "<table>" in html
        True

        Notes
        -----
        - Empty data returns minimal HTML with "No data" message
        - Missing values in rows are displayed as empty cells
        - All values are HTML-escaped for safety
        - Table includes alternating row colors (striped) for readability
        """
        try:
            # Handle empty data
            if not data:
                logger.warning("Empty data provided to table formatter")
                return self._build_empty_table_html()

            # Infer headers from first row if not provided
            if headers is None:
                headers = list(data[0].keys())

            # Build HTML table
            html_output = self._build_table_html(data, headers)

            logger.debug(
                f"Table HTML generated (rows={len(data)}, columns={len(headers)})"
            )

            return html_output

        except FormatterError:
            # Re-raise our own errors unchanged
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

    def _build_highlight_html(
        self,
        text: str,
        spans: list[tuple[int, int, str]],
        bias_types: dict[str, str],
    ) -> str:
        """Build HTML document with highlighted spans.

        Internal method that constructs the actual HTML with proper escaping
        and span wrapping.

        Parameters
        ----------
        text : str
            Original text to highlight
        spans : list[tuple[int, int, str]]
            Sorted list of (start, end, type) tuples
        bias_types : dict[str, str]
            Bias type to color mapping

        Returns
        -------
        str
            Complete HTML document
        """
        # Escape HTML in original text first (safety)
        escaped_text = html.escape(text)

        # Build CSS for bias types
        bias_css = self._build_bias_css(bias_types)

        # Apply spans to escaped text
        highlighted_content = self._apply_spans_to_text(escaped_text, spans, text)

        # Build complete HTML document
        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FairSense - Bias Analysis Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 900px;
            margin: 40px auto;
            padding: 30px;
            background-color: #f5f5f5;
            line-height: 1.8;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 15px;
            margin-top: 0;
        }}
        .text-content {{
            background-color: #fafafa;
            padding: 25px;
            border-radius: 6px;
            border-left: 5px solid #3498db;
            margin: 20px 0;
            font-size: 16px;
            line-height: 1.8;
        }}
        .legend {{
            margin-top: 30px;
            padding: 20px;
            background-color: #f9f9f9;
            border-radius: 6px;
            border: 1px solid #e0e0e0;
        }}
        .legend h2 {{
            margin-top: 0;
            color: #34495e;
            font-size: 18px;
        }}
        .legend-item {{
            display: inline-block;
            margin: 5px 10px 5px 0;
            padding: 5px 15px;
            border-radius: 4px;
            font-size: 14px;
        }}
        {bias_css}
        .bias-span {{
            padding: 2px 4px;
            border-radius: 3px;
            font-weight: 500;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Bias Analysis - Highlighted Text</h1>
        <div class="text-content">
            {highlighted_content}
        </div>
        <div class="legend">
            <h2>Bias Type Legend</h2>
            {self._build_legend(bias_types)}
        </div>
    </div>
</body>
</html>"""

    def _apply_spans_to_text(
        self,
        escaped_text: str,
        spans: list[tuple[int, int, str]],
        original_text: str,
    ) -> str:
        """Apply span highlights to escaped HTML text.

        This handles the tricky part: we need to apply spans based on character
        positions in the ORIGINAL text, but insert HTML into the ESCAPED text
        (which may have different character positions due to &lt; etc).

        Parameters
        ----------
        escaped_text : str
            HTML-escaped version of the text
        spans : list[tuple[int, int, str]]
            Spans with indices into original_text
        original_text : str
            Original unescaped text (for index reference)

        Returns
        -------
        str
            HTML with span elements inserted
        """
        if not spans:
            return escaped_text

        # For simplicity: work with original text, escape pieces, wrap spans
        # This avoids the complexity of mapping indices through HTML escaping
        result_parts = []
        last_end = 0

        for start, end, bias_type in spans:
            # Validate span indices
            if start < 0 or end > len(original_text) or start >= end:
                logger.warning(
                    f"Invalid span: start={start}, end={end}, text_len={len(original_text)}"
                )
                continue

            # Add text before span (escaped)
            if start > last_end:
                result_parts.append(html.escape(original_text[last_end:start]))

            # Add span with escaped content
            span_text = original_text[start:end]
            result_parts.append(
                f'<span class="bias-span bias-{bias_type}">{html.escape(span_text)}</span>'
            )

            last_end = end

        # Add remaining text (escaped)
        if last_end < len(original_text):
            result_parts.append(html.escape(original_text[last_end:]))

        return "".join(result_parts)

    def _build_bias_css(self, bias_types: dict[str, str]) -> str:
        """Build CSS rules for bias type classes.

        Parameters
        ----------
        bias_types : dict[str, str]
            Bias type to color mapping

        Returns
        -------
        str
            CSS rules as string
        """
        css_rules = []
        for bias_type, color in bias_types.items():
            css_rules.append(
                f"        .bias-{bias_type} {{ background-color: {color}; }}"
            )

        return "\n".join(css_rules)

    def _build_legend(self, bias_types: dict[str, str]) -> str:
        """Build HTML legend showing bias type colors.

        Parameters
        ----------
        bias_types : dict[str, str]
            Bias type to color mapping

        Returns
        -------
        str
            HTML for legend items
        """
        legend_items = []
        for bias_type, _color in bias_types.items():
            # Capitalize bias type for display
            display_name = bias_type.replace("_", " ").title()
            legend_items.append(
                f'<span class="legend-item bias-{bias_type}">{display_name}</span>'
            )

        return " ".join(legend_items)

    def _build_table_html(
        self,
        data: list[dict[str, Any]],
        headers: list[str],
    ) -> str:
        """Build HTML document with styled data table.

        Parameters
        ----------
        data : list[dict[str, Any]]
            Row data
        headers : list[str]
            Column headers

        Returns
        -------
        str
            Complete HTML document
        """
        # Build table rows
        rows_html = []
        for row in data:
            cells = []
            for header in headers:
                value = row.get(header, "")
                # Escape cell content for safety
                escaped_value = html.escape(str(value))
                cells.append(f"                <td>{escaped_value}</td>")
            rows_html.append(
                "            <tr>\n" + "\n".join(cells) + "\n            </tr>"
            )

        rows_html_str = "\n".join(rows_html)

        # Build header cells
        header_cells = [
            f"                <th>{html.escape(header)}</th>" for header in headers
        ]
        headers_html = "\n".join(header_cells)

        # Complete HTML document
        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FairSense - Data Table</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            background-color: white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }}
        th, td {{
            border: 1px solid #e0e0e0;
            padding: 14px 18px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 14px;
            letter-spacing: 0.5px;
        }}
        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        tr:hover {{
            background-color: #e3f2fd;
        }}
        td {{
            font-size: 14px;
            color: #2c3e50;
        }}
    </style>
</head>
<body>
    <table>
        <thead>
            <tr>
{headers_html}
            </tr>
        </thead>
        <tbody>
{rows_html_str}
        </tbody>
    </table>
</body>
</html>"""

    def _build_empty_table_html(self) -> str:
        """Build HTML for empty table case.

        Returns
        -------
        str
            Complete HTML document with "No data" message
        """
        return """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FairSense - Data Table</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }
        .no-data {
            background-color: white;
            padding: 40px;
            text-align: center;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            color: #7f8c8d;
            font-size: 18px;
        }
    </style>
</head>
<body>
    <div class="no-data">
        <p><strong>No data available</strong></p>
        <p>The table has no rows to display.</p>
    </div>
</body>
</html>"""
