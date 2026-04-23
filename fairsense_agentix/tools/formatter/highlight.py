"""HTML generation for bias span highlighting (fragments and full documents)."""

from __future__ import annotations

import html
import logging


logger = logging.getLogger(__name__)


def apply_spans_dark(
    original_text: str,
    spans: list[tuple[int, int, str]],
    bias_types: dict[str, str],
) -> str:
    """Apply span highlights with dark-mode inline styles."""
    if not spans:
        return html.escape(original_text)

    result_parts: list[str] = []
    last_end = 0

    for start, end, bias_type in spans:
        if start < 0 or end > len(original_text) or start >= end:
            logger.warning(
                f"Invalid span: start={start}, end={end}, "
                f"text_len={len(original_text)}",
            )
            continue

        if start > last_end:
            result_parts.append(html.escape(original_text[last_end:start]))

        span_text = original_text[start:end]
        color = bias_types.get(
            bias_type,
            "#FF6B9D",
        )  # Default pink for unknown types
        result_parts.append(
            f'<span style="background-color: {color}30; color: {color}; '
            f'padding: 2px 4px; border-radius: 3px; font-weight: 500;">'
            f"{html.escape(span_text)}</span>",
        )

        last_end = end

    if last_end < len(original_text):
        result_parts.append(html.escape(original_text[last_end:]))

    return "".join(result_parts)


def build_dark_fragment(
    text: str,
    spans: list[tuple[int, int, str]],
    bias_types: dict[str, str],
) -> str:
    """Build dark-mode HTML fragment (not complete document)."""
    highlighted_content = apply_spans_dark(text, spans, bias_types)
    _dark_style = (
        "font-family: ui-monospace, 'Cascadia Code', 'Source Code Pro', Menlo, "
        "Consolas, monospace; font-size: 14px; line-height: 1.6; color: #e5e7eb; "
        "background: transparent; padding: 16px; border-radius: 6px; "
        "white-space: pre-wrap; word-break: break-word;"
    )
    return f'<div style="{_dark_style}">\n{highlighted_content}\n</div>'


def apply_spans_to_text(
    escaped_text: str,
    spans: list[tuple[int, int, str]],
    original_text: str,
) -> str:
    """Apply span highlights; indices refer to original (unescaped) text."""
    if not spans:
        return escaped_text

    result_parts: list[str] = []
    last_end = 0

    for start, end, bias_type in spans:
        if start < 0 or end > len(original_text) or start >= end:
            logger.warning(
                f"Invalid span: start={start}, end={end}, "
                f"text_len={len(original_text)}",
            )
            continue

        if start > last_end:
            result_parts.append(html.escape(original_text[last_end:start]))

        span_text = original_text[start:end]
        result_parts.append(
            f'<span class="bias-span bias-{bias_type}">{html.escape(span_text)}</span>',
        )

        last_end = end

    if last_end < len(original_text):
        result_parts.append(html.escape(original_text[last_end:]))

    return "".join(result_parts)


def build_bias_css(bias_types: dict[str, str]) -> str:
    """Build CSS rules for bias type classes."""
    css_rules = []
    for bias_type, color in bias_types.items():
        css_rules.append(
            f"        .bias-{bias_type} {{ background-color: {color}; }}",
        )

    return "\n".join(css_rules)


def build_legend(bias_types: dict[str, str]) -> str:
    """Build HTML legend showing bias type colors."""
    legend_items = []
    for bias_type, _color in bias_types.items():
        display_name = bias_type.replace("_", " ").title()
        legend_items.append(
            f'<span class="legend-item bias-{bias_type}">{display_name}</span>',
        )

    return " ".join(legend_items)


def build_highlight_document(
    text: str,
    spans: list[tuple[int, int, str]],
    bias_types: dict[str, str],
) -> str:
    """Build complete HTML document with highlighted spans."""
    escaped_text = html.escape(text)
    bias_css = build_bias_css(bias_types)
    highlighted_content = apply_spans_to_text(escaped_text, spans, text)
    legend_html = build_legend(bias_types)

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
            {legend_html}
        </div>
    </div>
</body>
</html>"""
