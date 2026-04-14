"""Fake formatting and persistence tool implementations for testing."""

import csv
import html
import json
from pathlib import Path
from typing import Any


class FakeFormatterTool:
    """Fake formatting tool for testing.

    Generates simple HTML output.

    Examples
    --------
    >>> formatter = FakeFormatterTool()
    >>> html = formatter.highlight("Text", [(0, 4, "gender")], {"gender": "#FFB3BA"})
    >>> print(html[:20])
    '<!DOCTYPE html>...'
    """

    def __init__(self) -> None:
        """Initialize fake formatter."""
        self.call_count_highlight = 0
        self.call_count_table = 0

    def highlight(
        self,
        text: str,
        spans: list[tuple[int, int, str]],
        bias_types: dict[str, str],
    ) -> str:
        """Generate fake highlighted HTML."""
        self.call_count_highlight += 1

        phase_note = (
            "Note: This is a Phase 4 fake. "
            "Real highlighting will be implemented in Phase 5+."
        )
        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Bias Analysis - Highlighted Text</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 20px auto;
            padding: 20px;
            line-height: 1.6;
        }}
        .text-content {{
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 5px;
            border-left: 4px solid #4CAF50;
        }}
        .bias-gender {{ background-color: #FFB3BA; }}
        .bias-age {{ background-color: #FFDFBA; }}
        .bias-racial {{ background-color: #FFFFBA; }}
        .bias-disability {{ background-color: #BAE1FF; }}
        .bias-socioeconomic {{ background-color: #E0BBE4; }}
    </style>
</head>
<body>
    <h1>Bias Analysis - Highlighted Text</h1>
    <div class="text-content">
        <p>{html.escape(text)}</p>
    </div>
    <p><em>{phase_note}</em></p>
</body>
</html>"""

    def highlight_fragment(
        self,
        text: str,
        spans: list[tuple[int, int, str]],
        bias_types: dict[str, str],
    ) -> str:
        """Generate fake highlighted HTML fragment (not complete document).

        Returns an HTML fragment suitable for embedding in a larger document.
        This is the dark-mode compatible version without full HTML structure.
        """
        self.call_count_highlight += 1

        return (
            f'<div style="background-color: #1a1a1a; color: #e2e8f0; padding: 1rem; '
            f"border-radius: 0.5rem; font-family: 'Inter', sans-serif; "
            f'line-height: 1.6;">\n'
            f'    <p style="margin: 0;">{html.escape(text)}</p>\n'
            f'    <p style="margin-top: 1rem; font-size: 0.875rem; opacity: 0.7;">'
            f"<em>Note: This is a fake formatter for testing.</em></p>\n"
            f"</div>"
        )

    def table(
        self,
        data: list[dict[str, Any]],
        headers: list[str] | None = None,
    ) -> str:
        """Generate fake HTML table."""
        self.call_count_table += 1

        if not data:
            return "<table><tr><td>No data</td></tr></table>"

        if headers is None:
            headers = list(data[0].keys())

        html_doc = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        th {
            background-color: #4CAF50;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
    </style>
</head>
<body>
    <table>
        <thead>
            <tr>
"""

        for header in headers:
            html_doc += f"                <th>{html.escape(str(header))}</th>\n"
        html_doc += """            </tr>
        </thead>
        <tbody>
"""

        for row in data:
            html_doc += "            <tr>\n"
            for header in headers:
                value = row.get(header, "")
                html_doc += f"                <td>{html.escape(str(value))}</td>\n"
            html_doc += "            </tr>\n"

        html_doc += """        </tbody>
    </table>
</body>
</html>"""
        return html_doc


class FakePersistenceTool:
    """Fake persistence tool for testing.

    Simulates file writing without actually creating files (unless specified).

    Examples
    --------
    >>> persistence = FakePersistenceTool(output_dir=Path("outputs"))
    >>> path = persistence.save_csv([{"a": 1}], "test.csv")
    >>> print(path)
    outputs/test.csv
    """

    def __init__(self, output_dir: Path, *, actually_write: bool = False) -> None:
        """Initialize fake persistence tool.

        Parameters
        ----------
        output_dir : Path
            Output directory for saved files
        actually_write : bool, optional
            If True, actually write files (for integration tests),
            by default False
        """
        self.output_dir = output_dir
        self.actually_write = actually_write
        self.call_count_csv = 0
        self.call_count_json = 0
        self.saved_files: list[Path] = []

    def save_csv(
        self,
        data: list[dict[str, Any]],
        filename: str,
    ) -> Path:
        """Fake save CSV file."""
        self.call_count_csv += 1
        output_path = self.output_dir / filename

        if self.actually_write:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", newline="") as f:
                if data:
                    writer = csv.DictWriter(f, fieldnames=data[0].keys())
                    writer.writeheader()
                    writer.writerows(data)

        self.saved_files.append(output_path)
        return output_path.absolute()

    def save_json(
        self,
        data: Any,
        filename: str,
    ) -> Path:
        """Fake save JSON file."""
        self.call_count_json += 1
        output_path = self.output_dir / filename

        if self.actually_write:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with output_path.open("w") as f:
                json.dump(data, f, indent=2)

        self.saved_files.append(output_path)
        return output_path.absolute()
