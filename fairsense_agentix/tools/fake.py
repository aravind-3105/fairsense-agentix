"""Fake tool implementations for testing and development.

This module provides simple, deterministic fake implementations of all tool
protocols. These fakes are used for:
    - Development without expensive API calls or heavy dependencies
    - Testing workflows in isolation
    - Demonstrating system functionality (Phase 2-4)

All fake tools:
    - Return deterministic outputs (same input -> same output)
    - Track call counts and arguments for testing
    - Support custom return values for specific test scenarios
    - Satisfy their respective protocol interfaces

Examples
--------
    >>> from fairsense_agentix.tools.fake import FakeOCRTool
    >>> ocr = FakeOCRTool()
    >>> text = ocr.extract(b"image_data")
    >>> print(ocr.call_count)  # Track calls
    1
"""

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np


if TYPE_CHECKING:
    from fairsense_agentix.tools.interfaces import EmbedderTool


# ============================================================================
# Image Processing Fakes
# ============================================================================


class FakeOCRTool:
    """Fake OCR tool for testing.

    Returns deterministic text based on image size. Optionally accepts
    custom return value for specific test scenarios.

    Examples
    --------
    >>> ocr = FakeOCRTool()
    >>> text = ocr.extract(b"fake_image", language="eng")
    >>> print(text)
    'Extracted text from 10 byte image (language=eng)'

    >>> # Custom return value
    >>> ocr = FakeOCRTool(return_text="Custom OCR result")
    >>> ocr.extract(b"image")
    'Custom OCR result'
    """

    def __init__(self, *, return_text: str | None = None) -> None:
        """Initialize fake OCR with optional fixed return value.

        Parameters
        ----------
        return_text : str | None, optional
            Fixed text to return (if None, generate deterministic output),
            by default None
        """
        self.return_text = return_text
        self.call_count = 0
        self.last_call_args: dict[str, Any] = {}

    def extract(
        self,
        image_bytes: bytes,
        language: str = "eng",
        confidence_threshold: float = 0.5,
    ) -> str:
        """Extract fake text from image bytes."""
        self.call_count += 1
        self.last_call_args = {
            "image_bytes_len": len(image_bytes),
            "language": language,
            "confidence_threshold": confidence_threshold,
        }

        if self.return_text is not None:
            return self.return_text

        # Deterministic output based on input
        return (
            f"Extracted text from {len(image_bytes)} byte image (language={language})"
        )


class FakeCaptionTool:
    """Fake image captioning tool for testing.

    Returns deterministic captions based on image size.

    Examples
    --------
    >>> captioner = FakeCaptionTool()
    >>> caption = captioner.caption(b"image_data", max_length=50)
    >>> print(caption)
    'A photo or illustration (10 bytes)'
    """

    def __init__(self, *, return_caption: str | None = None) -> None:
        """Initialize fake captioner with optional fixed return value."""
        self.return_caption = return_caption
        self.call_count = 0
        self.last_call_args: dict[str, Any] = {}

    def caption(
        self,
        image_bytes: bytes,
        max_length: int = 100,
    ) -> str:
        """Generate fake caption for image."""
        self.call_count += 1
        self.last_call_args = {
            "image_bytes_len": len(image_bytes),
            "max_length": max_length,
        }

        if self.return_caption is not None:
            return self.return_caption

        # Deterministic caption
        caption = f"A photo or illustration ({len(image_bytes)} bytes)"
        return caption[:max_length]


# ============================================================================
# Text Generation Fakes
# ============================================================================


class FakeLLMTool:
    """Fake LLM tool for testing.

    Returns deterministic bias analysis text. Simulates token counting and
    tracks calls for testing.

    Examples
    --------
    >>> llm = FakeLLMTool()
    >>> result = llm.predict("Analyze this text for bias")
    >>> print(result[:50])
    '**Bias Analysis Report**...'
    >>> print(llm.get_token_count("Sample text"))
    2
    """

    def __init__(self, *, return_text: str | None = None) -> None:
        """Initialize fake LLM with optional fixed return value."""
        self.return_text = return_text
        self.call_count = 0
        self.last_call_args: dict[str, Any] = {}

    def predict(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        timeout: int = 60,
    ) -> str:
        """Generate fake LLM prediction.

        Returns structured JSON matching BiasAnalysisOutput schema for
        deterministic behavior that mimics real LLM structured outputs.
        """
        self.call_count += 1
        self.last_call_args = {
            "prompt_len": len(prompt),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
        }

        if self.return_text is not None:
            return self.return_text

        # Generate deterministic structured output matching BiasAnalysisOutput
        # Use hash of prompt for deterministic but varied outputs
        import hashlib  # noqa: PLC0415

        prompt_hash = int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16)
        seed = prompt_hash % 100 / 100.0  # Deterministic seed in [0, 1)

        return json.dumps(
            {
                "bias_detected": True,
                "bias_instances": [
                    {
                        "type": "age",
                        "severity": "medium",
                        "text_span": "sample text",
                        "visual_element": "",
                        "explanation": f"Deterministic fake bias instance (seed={seed:.2f})",
                        "start_char": 0,
                        "end_char": 11,
                        "evidence_source": "text",
                    }
                ],
                "overall_assessment": (
                    "Fake LLM deterministic output. The text demonstrates "
                    f"potential bias patterns (analysis seed={seed:.2f})."
                ),
                "risk_level": "medium",
            },
            indent=2,
        )

    def get_token_count(self, text: str) -> int:
        """Estimate token count (simple word-based approximation)."""
        return len(text.split())


class FakeSummarizerTool:
    """Fake summarization tool for testing.

    Returns deterministic summaries based on input text.

    Examples
    --------
    >>> summarizer = FakeSummarizerTool()
    >>> summary = summarizer.summarize("Long text here...", max_length=100)
    >>> print(summary)
    '**Summary**: Minimal bias detected...'
    """

    def __init__(self, *, return_summary: str | None = None) -> None:
        """Initialize fake summarizer with optional fixed return value."""
        self.return_summary = return_summary
        self.call_count = 0
        self.last_call_args: dict[str, Any] = {}

    def summarize(
        self,
        text: str,
        max_length: int = 200,
    ) -> str:
        """Generate fake summary."""
        self.call_count += 1
        self.last_call_args = {
            "text_len": len(text),
            "max_length": max_length,
        }

        if self.return_summary is not None:
            # Truncate custom summary to max_length (consistent with real)
            return self.return_summary[:max_length]

        # Deterministic summary
        summary = (
            "**Summary**: Minimal bias detected. Text is generally inclusive with\n"
            "neutral language across gender, age, and racial dimensions. Consider adding explicit\n"
            "diversity statements for improvement. Confidence: 0.85"
        )
        return summary[:max_length]


# ============================================================================
# Embedding & Search Fakes
# ============================================================================


class FakeEmbedderTool:
    """Fake embedding tool for testing.

    Returns deterministic vectors based on text hash.

    Examples
    --------
    >>> embedder = FakeEmbedderTool(dimension=384)
    >>> vector = embedder.encode("Sample text")
    >>> len(vector)
    384
    >>> embedder.dimension
    384
    """

    def __init__(self, dimension: int = 384) -> None:
        """Initialize fake embedder with specified dimension."""
        self._dimension = dimension
        self.call_count = 0
        self.last_call_args: dict[str, Any] = {}

    def encode(self, text: str) -> np.ndarray:
        """Generate fake embedding vector."""
        self.call_count += 1
        self.last_call_args = {"text_len": len(text)}

        # Generate deterministic vector based on text hash
        # Use hash to get different but consistent vectors for different texts
        text_hash = hash(text)
        vector = [(text_hash + i) % 1000 / 1000.0 for i in range(self._dimension)]
        return np.array(vector, dtype="float32")

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension


class FakeFAISSIndexTool:
    """Fake FAISS index tool for testing.

    Returns deterministic search results. Optionally accepts custom results
    for specific test scenarios.

    Examples
    --------
    >>> index = FakeFAISSIndexTool(index_path=Path("data/risks.faiss"))
    >>> results = index.search([0.1, 0.2, ...], top_k=3)
    >>> len(results)
    3
    >>> results[0]["text"]
    'Risk 1: Bias in training data'
    """

    def __init__(
        self,
        index_path: Path,
        embedder: "EmbedderTool | None" = None,
        *,
        return_results: list[dict[str, Any]] | None = None,
    ) -> None:
        """Initialize fake FAISS index.

        Parameters
        ----------
        index_path : Path
            Path to fake index file
        embedder : EmbedderTool | None, optional
            Embedder for text-based search, by default None
        return_results : list[dict[str, Any]] | None, optional
            Fixed results to return (if None, generate deterministic results),
            by default None
        """
        self._index_path = index_path
        self.embedder = embedder
        self.return_results = return_results
        self.call_count = 0
        self.last_call_args: dict[str, Any] = {}

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Perform fake vector search."""
        self.call_count += 1
        self.last_call_args = {
            "query_vector_dim": len(query_vector),
            "top_k": top_k,
        }

        if self.return_results is not None:
            return self.return_results[:top_k]

        # Generate deterministic fake results
        # Use vector sum as seed for deterministic but varied results
        vector_sum = sum(query_vector)
        results = []

        for i in range(top_k):
            results.append(
                {
                    "id": f"item_{i}",
                    "text": f"Risk {i + 1}: Bias in training data (seed={vector_sum:.2f})",
                    "score": 0.95 - (i * 0.1),  # Descending scores
                    "severity": "high" if i < 2 else "medium",
                    "category": "fairness",
                }
            )

        return results

    def search_by_text(
        self,
        query_text: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Perform fake text-based search."""
        embedder = FakeEmbedderTool() if self.embedder is None else self.embedder
        query_vector = embedder.encode(query_text)
        # Convert numpy array to list for search() protocol compatibility
        return self.search(query_vector.tolist(), top_k)

    @property
    def index_path(self) -> Path:
        """Get index path."""
        return self._index_path


# ============================================================================
# Formatting & Persistence Fakes
# ============================================================================


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

        # Simple HTML with styles
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
        <p>{text}</p>
    </div>
    <p><em>Note: This is a Phase 4 fake. Real highlighting will be implemented in Phase 5+.</em></p>
</body>
</html>"""

    def table(
        self,
        data: list[dict[str, Any]],
        headers: list[str] | None = None,
    ) -> str:
        """Generate fake HTML table."""
        self.call_count_table += 1

        if not data:
            return "<table><tr><td>No data</td></tr></table>"

        # Infer headers from first row if not provided
        if headers is None:
            headers = list(data[0].keys())

        # Build HTML table
        html = """<!DOCTYPE html>
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

        # Add headers
        for header in headers:
            html += f"                <th>{header}</th>\n"
        html += """            </tr>
        </thead>
        <tbody>
"""

        # Add data rows
        for row in data:
            html += "            <tr>\n"
            for header in headers:
                value = row.get(header, "")
                html += f"                <td>{value}</td>\n"
            html += "            </tr>\n"

        html += """        </tbody>
    </table>
</body>
</html>"""
        return html


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
