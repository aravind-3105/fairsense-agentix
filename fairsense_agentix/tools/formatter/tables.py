"""HTML generation for structured data tables."""

from __future__ import annotations

import html
from typing import Any


def build_table_html(
    data: list[dict[str, Any]],
    headers: list[str],
) -> str:
    """Build HTML document with styled data table."""
    rows_html = []
    for row in data:
        cells = []
        for header in headers:
            value = row.get(header, "")
            escaped_value = html.escape(str(value))
            cells.append(f"                <td>{escaped_value}</td>")
        rows_html.append(
            "            <tr>\n" + "\n".join(cells) + "\n            </tr>",
        )

    rows_html_str = "\n".join(rows_html)

    header_cells = [
        f"                <th>{html.escape(header)}</th>" for header in headers
    ]
    headers_html = "\n".join(header_cells)

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


def build_empty_table_html() -> str:
    """Build HTML for empty table case."""
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
