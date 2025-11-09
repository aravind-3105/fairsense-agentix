"""Tool protocol interfaces for FairSense-AgentiX.

This module defines Protocol classes (abstract interfaces) that all tool
implementations must satisfy. Protocols enable:
    - Structural subtyping (duck typing with type checking)
    - Swappable implementations (fake, real, mock)
    - Clear contracts for tool behavior
    - Type safety without inheritance requirements

Protocols use Python's typing.Protocol with @runtime_checkable decorator,
allowing isinstance() checks at runtime.

Examples
--------
    >>> from fairsense_agentix.tools.interfaces import OCRTool
    >>> class MyOCR:
    ...     def extract(self, image_bytes: bytes, **kwargs) -> str:
    ...         return "extracted text"
    >>> tool = MyOCR()
    >>> isinstance(tool, OCRTool)  # True - structural subtyping
    True
"""

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np


# ============================================================================
# Image Processing Tools
# ============================================================================


@runtime_checkable
class OCRTool(Protocol):
    """Protocol for optical character recognition (OCR) tools.

    Extracts text from images using various OCR engines (Tesseract, PaddleOCR,
    etc.). Implementations should handle common image formats (PNG, JPEG, etc.).

    Examples
    --------
    >>> ocr = TesseractOCR()
    >>> text = ocr.extract(image_bytes, language="eng")
    >>> print(text)
    'Extracted text from image'
    """

    def extract(
        self,
        image_bytes: bytes,
        language: str = "eng",
        confidence_threshold: float = 0.5,
    ) -> str:
        """Extract text from image bytes.

        Parameters
        ----------
        image_bytes : bytes
            Raw image data (PNG, JPEG, etc.)
        language : str, optional
            OCR language code (e.g., 'eng', 'fra', 'spa'), by default "eng"
        confidence_threshold : float, optional
            Minimum confidence for including detected text (0.0-1.0),
            by default 0.5

        Returns
        -------
        str
            Extracted text from image

        Raises
        ------
        OCRError
            If extraction fails (invalid image, unsupported format, etc.)
        """
        ...


@runtime_checkable
class CaptionTool(Protocol):
    """Protocol for image captioning tools.

    Generates natural language descriptions of images using vision-language
    models (BLIP, BLIP2, LLAVA, etc.).

    Examples
    --------
    >>> captioner = BLIP2Caption()
    >>> caption = captioner.caption(image_bytes, max_length=100)
    >>> print(caption)
    'A person sitting at a desk with a laptop'
    """

    def caption(
        self,
        image_bytes: bytes,
        max_length: int = 100,
    ) -> str:
        """Generate natural language caption for image.

        Parameters
        ----------
        image_bytes : bytes
            Raw image data (PNG, JPEG, etc.)
        max_length : int, optional
            Maximum caption length in characters, by default 100

        Returns
        -------
        str
            Natural language description of image content

        Raises
        ------
        CaptionError
            If captioning fails (invalid image, model error, etc.)
        """
        ...


# ============================================================================
# Text Generation Tools
# ============================================================================


@runtime_checkable
class LLMTool(Protocol):
    """Protocol for large language model (LLM) tools.

    Generates text completions from prompts using various LLM providers
    (OpenAI, Anthropic, local models, etc.).

    Examples
    --------
    >>> llm = OpenAILLM(model="gpt-4")
    >>> result = llm.predict("Analyze this text for bias: ...", temperature=0.3)
    >>> print(result)
    '**Bias Analysis**: The text contains...'
    """

    def predict(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        timeout: int = 60,
    ) -> str:
        """Generate text completion from prompt.

        Parameters
        ----------
        prompt : str
            Input prompt for the LLM
        temperature : float, optional
            Sampling temperature (0.0 = deterministic, 1.0 = creative),
            by default 0.3
        max_tokens : int, optional
            Maximum tokens to generate, by default 2000
        timeout : int, optional
            Request timeout in seconds, by default 60

        Returns
        -------
        str
            Generated text completion

        Raises
        ------
        LLMError
            If prediction fails (timeout, rate limit, API error, etc.)
        """
        ...

    def get_token_count(self, text: str) -> int:
        """Estimate token count for text.

        Used for cost tracking and prompt sizing.

        Parameters
        ----------
        text : str
            Text to count tokens for

        Returns
        -------
        int
            Estimated number of tokens
        """
        ...


@runtime_checkable
class SummarizerTool(Protocol):
    """Protocol for text summarization tools.

    Condenses long text into shorter summaries. Often implemented using LLMs
    but can also use extractive summarization methods.

    Examples
    --------
    >>> summarizer = LLMSummarizer()
    >>> summary = summarizer.summarize(long_text, max_length=200)
    >>> print(summary)
    'Summary: The text discusses...'
    """

    def summarize(
        self,
        text: str,
        max_length: int = 200,
    ) -> str:
        """Condense text into shorter summary.

        Parameters
        ----------
        text : str
            Text to summarize
        max_length : int, optional
            Maximum summary length in characters, by default 200

        Returns
        -------
        str
            Condensed summary of input text

        Raises
        ------
        LLMError
            If summarization fails
        """
        ...


# ============================================================================
# Embedding & Search Tools
# ============================================================================


@runtime_checkable
class EmbedderTool(Protocol):
    """Protocol for text embedding tools.

    Converts text into dense vector representations using embedding models
    (sentence-transformers, OpenAI embeddings, etc.).

    Examples
    --------
    >>> embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")
    >>> vector = embedder.encode("Sample text")
    >>> print(len(vector))
    384
    >>> print(embedder.dimension)
    384
    """

    def encode(self, text: str) -> np.ndarray:
        """Convert text to dense vector embedding.

        Parameters
        ----------
        text : str
            Text to embed

        Returns
        -------
        np.ndarray
            Dense vector representation (1D array of shape (dimension,))

        Raises
        ------
        EmbeddingError
            If embedding fails (text too long, model error, etc.)
        """
        ...

    @property
    def dimension(self) -> int:
        """Get embedding dimension.

        Returns
        -------
        int
            Vector dimension (e.g., 384 for all-MiniLM-L6-v2)
        """
        ...


@runtime_checkable
class FAISSIndexTool(Protocol):
    """Protocol for FAISS vector similarity search tools.

    Performs efficient nearest-neighbor search over large vector collections.
    Used for retrieving relevant risks and RMF recommendations.

    Examples
    --------
    >>> index = FAISSIndex(index_path="data/indexes/risks.faiss")
    >>> results = index.search(query_vector, top_k=5)
    >>> for result in results:
    ...     print(result["text"], result["score"])
    """

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Search index for nearest neighbors.

        Parameters
        ----------
        query_vector : list[float]
            Query vector to search for
        top_k : int, optional
            Number of results to return, by default 5

        Returns
        -------
        list[dict[str, Any]]
            List of result dictionaries with keys:
                - "id": int or str
                - "text": str
                - "score": float (similarity score)
                - Additional metadata fields

        Raises
        ------
        FAISSError
            If search fails (index not found, dimension mismatch, etc.)
        """
        ...

    def search_by_text(
        self,
        query_text: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Search using text with automatic embedding.

        Requires the index to have an associated embedder.

        Parameters
        ----------
        query_text : str
            Text query to search for
        top_k : int, optional
            Number of results to return, by default 5

        Returns
        -------
        list[dict[str, Any]]
            List of result dictionaries (same format as search())

        Raises
        ------
        FAISSError
            If search fails
        """
        ...

    @property
    def index_path(self) -> Path:
        """Get path to FAISS index file.

        Returns
        -------
        Path
            Path to .faiss index file
        """
        ...


# ============================================================================
# Formatting & Persistence Tools
# ============================================================================


@runtime_checkable
class FormatterTool(Protocol):
    """Protocol for output formatting tools.

    Generates formatted output (HTML with highlights, tables) from structured data.

    Examples
    --------
    >>> formatter = HTMLFormatter()
    >>> html = formatter.highlight(
    ...     text="Job posting text",
    ...     spans=[(0, 3, "gender")],
    ...     bias_types={"gender": "#FFB3BA"},
    ... )
    >>> table = formatter.table(
    ...     data=[{"risk": "Bias", "severity": "High"}], headers=["Risk", "Severity"]
    ... )
    """

    def highlight(
        self,
        text: str,
        spans: list[tuple[int, int, str]],
        bias_types: dict[str, str],
    ) -> str:
        """Generate HTML with highlighted text spans.

        Parameters
        ----------
        text : str
            Text to highlight
        spans : list[tuple[int, int, str]]
            List of (start_idx, end_idx, bias_type) tuples
        bias_types : dict[str, str]
            Mapping of bias_type to color (e.g., {"gender": "#FFB3BA"})

        Returns
        -------
        str
            HTML string with highlighted spans

        Raises
        ------
        FormatterError
            If formatting fails (invalid spans, template error, etc.)
        """
        ...

    def table(
        self,
        data: list[dict[str, Any]],
        headers: list[str] | None = None,
    ) -> str:
        """Generate HTML table from structured data.

        Parameters
        ----------
        data : list[dict[str, Any]]
            List of row dictionaries
        headers : list[str] | None, optional
            Column headers (if None, infer from first row keys), by default None

        Returns
        -------
        str
            HTML table string

        Raises
        ------
        FormatterError
            If formatting fails (empty data, invalid structure, etc.)
        """
        ...


@runtime_checkable
class PersistenceTool(Protocol):
    """Protocol for file persistence tools.

    Saves data to disk in various formats (CSV, JSON, etc.).

    Examples
    --------
    >>> persistence = FileWriter(output_dir="outputs")
    >>> csv_path = persistence.save_csv(
    ...     data=[{"risk": "Bias", "severity": "High"}], filename="risks.csv"
    ... )
    >>> print(csv_path)
    /absolute/path/to/outputs/risks.csv
    """

    def save_csv(
        self,
        data: list[dict[str, Any]],
        filename: str,
    ) -> Path:
        """Save data to CSV file.

        Parameters
        ----------
        data : list[dict[str, Any]]
            List of row dictionaries
        filename : str
            Output filename (e.g., "risks.csv")

        Returns
        -------
        Path
            Absolute path to saved CSV file

        Raises
        ------
        PersistenceError
            If save fails (permission denied, disk full, etc.)
        """
        ...

    def save_json(
        self,
        data: Any,
        filename: str,
    ) -> Path:
        """Save data to JSON file.

        Parameters
        ----------
        data : Any
            JSON-serializable data
        filename : str
            Output filename (e.g., "result.json")

        Returns
        -------
        Path
            Absolute path to saved JSON file

        Raises
        ------
        PersistenceError
            If save fails or data not JSON-serializable
        """
        ...
