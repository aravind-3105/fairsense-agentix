"""Tests for formatter tools.

This module tests the formatter tool implementations, including protocol
compliance, fake tool behavior, registry resolution, and real tool functionality.
"""

import pytest

from fairsense_agentix.tools.exceptions import FormatterError
from fairsense_agentix.tools.fake import FakeFormatterTool
from fairsense_agentix.tools.formatter import HTMLFormatter
from fairsense_agentix.tools.interfaces import FormatterTool


# ============================================================================
# Protocol Compliance Tests
# ============================================================================


class TestProtocolCompliance:
    """Test that formatter implementations satisfy the protocol."""

    def test_fake_formatter_satisfies_protocol(self):
        """Verify FakeFormatterTool implements FormatterTool protocol."""
        fake_formatter = FakeFormatterTool()
        assert isinstance(fake_formatter, FormatterTool)

    def test_html_formatter_satisfies_protocol(self):
        """Verify HTMLFormatter implements FormatterTool protocol."""
        formatter = HTMLFormatter()
        assert isinstance(formatter, FormatterTool)


# ============================================================================
# Fake Formatter Tests
# ============================================================================


class TestFakeFormatter:
    """Test fake formatter for unit testing and development."""

    def test_highlight_returns_html(self):
        """Test fake formatter highlight returns valid HTML."""
        formatter = FakeFormatterTool()

        html = formatter.highlight(
            text="Sample text",
            spans=[(0, 6, "gender")],
            bias_types={"gender": "#FFB3BA"},
        )

        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html
        assert "Sample text" in html

    def test_table_returns_html(self):
        """Test fake formatter table returns valid HTML."""
        formatter = FakeFormatterTool()

        data = [{"col1": "value1", "col2": "value2"}]
        html = formatter.table(data)

        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html
        assert "<table>" in html

    def test_highlight_call_tracking(self):
        """Test fake formatter tracks highlight calls."""
        formatter = FakeFormatterTool()

        assert formatter.call_count_highlight == 0

        formatter.highlight("Text", [], {})
        assert formatter.call_count_highlight == 1

        formatter.highlight("Text 2", [], {})
        assert formatter.call_count_highlight == 2

    def test_table_call_tracking(self):
        """Test fake formatter tracks table calls."""
        formatter = FakeFormatterTool()

        assert formatter.call_count_table == 0

        formatter.table([{"a": 1}])
        assert formatter.call_count_table == 1

        formatter.table([{"b": 2}])
        assert formatter.call_count_table == 2


# ============================================================================
# HTML Formatter - Highlight Tests
# ============================================================================


class TestHTMLFormatterHighlight:
    """Test HTML formatter highlight functionality."""

    def test_basic_highlight(self):
        """Test basic span highlighting."""
        formatter = HTMLFormatter()

        text = "Looking for a young developer"
        spans = [(14, 19, "age")]  # "young"
        bias_types = {"age": "#FFDFBA"}

        html = formatter.highlight(text, spans, bias_types)

        # Verify structure
        assert "<!DOCTYPE html>" in html
        assert "<html>" in html
        assert "young" in html
        assert 'class="bias-age"' in html or 'class="bias-span bias-age"' in html

    def test_highlight_with_multiple_spans(self):
        """Test highlighting multiple non-overlapping spans."""
        formatter = HTMLFormatter()

        text = "We need a young rockstar developer"
        spans = [
            (10, 15, "age"),  # "young"
            (16, 24, "gender"),  # "rockstar"
        ]
        bias_types = {
            "age": "#FFDFBA",
            "gender": "#FFB3BA",
        }

        html = formatter.highlight(text, spans, bias_types)

        assert "young" in html
        assert "rockstar" in html
        assert "bias-age" in html
        assert "bias-gender" in html

    def test_highlight_empty_spans(self):
        """Test highlighting with no spans."""
        formatter = HTMLFormatter()

        text = "Clean text with no bias"
        spans = []
        bias_types = {}

        html = formatter.highlight(text, spans, bias_types)

        assert "<!DOCTYPE html>" in html
        assert "Clean text with no bias" in html

    def test_highlight_html_escaping(self):
        """Test HTML characters in text are properly escaped."""
        formatter = HTMLFormatter()

        text = "Code: if (x < 5 && y > 10) { return true; }"
        spans = []
        bias_types = {}

        html = formatter.highlight(text, spans, bias_types)

        # HTML should be escaped (< becomes &lt;, > becomes &gt;, & becomes &amp;)
        assert "&lt;" in html
        assert "&gt;" in html
        assert "&amp;" in html
        # Raw HTML characters should NOT appear in content
        assert "if (x < 5 &&" not in html  # This would break HTML

    def test_highlight_with_custom_colors(self):
        """Test custom color mapping."""
        color_map = {"custom_bias": "#FF0000"}
        formatter = HTMLFormatter(color_map=color_map)

        text = "Custom bias type"
        spans = [(0, 6, "custom_bias")]
        bias_types = {"custom_bias": "#FF0000"}

        html = formatter.highlight(text, spans, bias_types)

        assert "bias-custom_bias" in html
        assert "#FF0000" in html

    def test_highlight_invalid_span_indices(self):
        """Test handling of invalid span indices."""
        formatter = HTMLFormatter()

        text = "Short text"
        spans = [
            (0, 5, "valid"),  # Valid
            (-1, 3, "negative_start"),  # Invalid: negative start
            (5, 100, "out_of_bounds"),  # Invalid: end > len(text)
            (8, 5, "backwards"),  # Invalid: start >= end
        ]
        bias_types = {"valid": "#FFB3BA"}

        # Should not raise error, just skip invalid spans
        html = formatter.highlight(text, spans, bias_types)

        assert "<!DOCTYPE html>" in html
        # Text content present (may be split by spans)
        assert "Short" in html
        assert "text" in html

    def test_highlight_overlapping_spans(self):
        """Test handling of overlapping spans."""
        formatter = HTMLFormatter()

        text = "This is problematic text"
        spans = [
            (8, 19, "type1"),  # "problematic"
            (12, 24, "type2"),  # "ematic text" (overlaps with previous)
        ]
        bias_types = {
            "type1": "#FFB3BA",
            "type2": "#FFDFBA",
        }

        # Should handle gracefully (implementation may apply first span)
        html = formatter.highlight(text, spans, bias_types)

        assert "<!DOCTYPE html>" in html
        assert "problematic" in html or "ematic" in html

    def test_highlight_includes_legend(self):
        """Test that highlight HTML includes bias type legend."""
        formatter = HTMLFormatter()

        text = "Sample text"
        spans = [(0, 6, "gender")]
        bias_types = {"gender": "#FFB3BA"}

        html = formatter.highlight(text, spans, bias_types)

        # Should include legend section
        assert "legend" in html.lower()
        assert "Gender" in html  # Capitalized bias type name

    def test_highlight_non_string_text_raises_error(self):
        """Test non-string text raises FormatterError."""
        formatter = HTMLFormatter()

        with pytest.raises(FormatterError) as exc_info:
            formatter.highlight(
                text=123,  # Invalid: not a string
                spans=[],
                bias_types={},
            )

        assert "string" in str(exc_info.value).lower()
        assert "text_type" in exc_info.value.context


# ============================================================================
# HTML Formatter - Table Tests
# ============================================================================


class TestHTMLFormatterTable:
    """Test HTML formatter table functionality."""

    def test_basic_table(self):
        """Test basic table generation."""
        formatter = HTMLFormatter()

        data = [
            {"risk": "Bias", "severity": "High", "score": 0.89},
            {"risk": "Privacy", "severity": "Medium", "score": 0.65},
        ]
        headers = ["risk", "severity", "score"]

        html = formatter.table(data, headers)

        # Verify structure
        assert "<!DOCTYPE html>" in html
        assert "<table>" in html
        assert "<thead>" in html
        assert "<tbody>" in html
        assert "Bias" in html
        assert "Privacy" in html
        assert "High" in html
        assert "Medium" in html

    def test_table_infers_headers(self):
        """Test table infers headers from first row if not provided."""
        formatter = HTMLFormatter()

        data = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
        ]

        html = formatter.table(data, headers=None)

        # Should infer headers from first row keys
        assert "name" in html
        assert "age" in html
        assert "Alice" in html
        assert "Bob" in html

    def test_table_empty_data(self):
        """Test table with empty data list."""
        formatter = HTMLFormatter()

        html = formatter.table(data=[], headers=None)

        # Should return minimal HTML with "No data" message
        assert "<!DOCTYPE html>" in html
        assert "no data" in html.lower() or "empty" in html.lower()

    def test_table_missing_values(self):
        """Test table handles missing values in rows."""
        formatter = HTMLFormatter()

        data = [
            {"col1": "value1", "col2": "value2", "col3": "value3"},
            {"col1": "value4", "col3": "value6"},  # Missing col2
        ]
        headers = ["col1", "col2", "col3"]

        html = formatter.table(data, headers)

        # Should handle missing value gracefully (empty cell)
        assert "value1" in html
        assert "value4" in html
        assert "value6" in html

    def test_table_html_escaping(self):
        """Test HTML characters in table data are escaped."""
        formatter = HTMLFormatter()

        data = [
            {"code": "<script>alert('xss')</script>", "safe": "normal text"},
        ]
        headers = ["code", "safe"]

        html = formatter.table(data, headers)

        # HTML should be escaped
        assert "&lt;script&gt;" in html
        assert (
            "<script>" not in html or html.count("<script>") <= 1
        )  # Only in <script> tag for CSS

    def test_table_with_special_characters(self):
        """Test table handles special characters."""
        formatter = HTMLFormatter()

        data = [
            {"text": "Symbols: & < > \" '", "emoji": "🎉 ✨"},
        ]
        headers = ["text", "emoji"]

        html = formatter.table(data, headers)

        # Should escape HTML entities but preserve Unicode
        assert "&amp;" in html  # & escaped
        assert "&lt;" in html  # < escaped
        assert "&gt;" in html  # > escaped
        # Emoji may or may not be present depending on encoding

    def test_table_includes_styling(self):
        """Test table HTML includes CSS styling."""
        formatter = HTMLFormatter()

        data = [{"col": "value"}]
        html = formatter.table(data)

        # Should include CSS
        assert "<style>" in html
        assert "border-collapse" in html
        assert "background-color" in html

    def test_table_striped_rows(self):
        """Test table includes alternating row colors."""
        formatter = HTMLFormatter()

        data = [
            {"id": 1},
            {"id": 2},
            {"id": 3},
        ]
        html = formatter.table(data)

        # Should include CSS for striped rows
        assert "nth-child(even)" in html or "odd" in html.lower()


# ============================================================================
# Golden HTML Tests
# ============================================================================


class TestGoldenHTML:
    """Test HTML output against expected golden outputs."""

    def test_highlight_golden_output_structure(self):
        """Test highlight produces expected HTML structure."""
        formatter = HTMLFormatter()

        text = "Sample text"
        spans = [(0, 6, "gender")]
        bias_types = {"gender": "#FFB3BA"}

        html = formatter.highlight(text, spans, bias_types)

        # Golden structure checks
        assert html.startswith("<!DOCTYPE html>")
        assert "<html>" in html
        assert "<head>" in html
        assert '<meta charset="UTF-8">' in html
        assert "<title>" in html
        assert "<style>" in html
        assert "<body>" in html
        assert "</body>" in html
        assert "</html>" in html

    def test_table_golden_output_structure(self):
        """Test table produces expected HTML structure."""
        formatter = HTMLFormatter()

        data = [{"col1": "value1"}]
        html = formatter.table(data)

        # Golden structure checks
        assert html.startswith("<!DOCTYPE html>")
        assert "<html>" in html
        assert "<table>" in html
        assert "<thead>" in html
        assert "<tbody>" in html
        assert "</tbody>" in html
        assert "</table>" in html

    def test_highlight_golden_span_wrapping(self):
        """Test highlight wraps spans correctly."""
        formatter = HTMLFormatter()

        text = "Test"
        spans = [(0, 4, "type1")]
        bias_types = {"type1": "#FF0000"}

        html = formatter.highlight(text, spans, bias_types)

        # Should wrap "Test" in a span with bias-type1 class
        assert '<span class="bias-span bias-type1">Test</span>' in html

    def test_table_golden_header_rendering(self):
        """Test table renders headers correctly."""
        formatter = HTMLFormatter()

        data = [{"Header1": "value"}]
        headers = ["Header1"]
        html = formatter.table(data, headers)

        # Should render header in <th> tag
        assert "<th>Header1</th>" in html


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_highlight_error_includes_context(self):
        """Test highlight errors include helpful context."""
        formatter = HTMLFormatter()

        with pytest.raises(FormatterError) as exc_info:
            # Force an error with invalid input
            formatter.highlight(
                text=None,  # Invalid
                spans=[],
                bias_types={},
            )

        # Error should have context
        error = exc_info.value
        assert error.context is not None

    def test_table_error_includes_context(self):
        """Test table errors include helpful context."""
        formatter = HTMLFormatter()

        # Create data that will cause an error during iteration
        class BadData:
            def __iter__(self):
                raise ValueError("Cannot iterate")

        with pytest.raises(FormatterError) as exc_info:
            # Force an error with data that fails during iteration
            formatter.table(
                data=BadData(),  # type: ignore
                headers=None,
            )

        # Error should have context
        error = exc_info.value
        assert error.context is not None
