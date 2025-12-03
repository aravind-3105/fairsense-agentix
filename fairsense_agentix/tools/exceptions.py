"""Custom exceptions for FairSense-AgentiX tool system.

This module defines a hierarchy of exceptions that provide clear, typed error
handling for tool operations. Each exception type corresponds to a specific
tool category, enabling precise error handling and debugging.

Examples
--------
    >>> from fairsense_agentix.tools.exceptions import OCRError
    >>> try:
    ...     result = ocr_tool.extract(image_bytes)
    ... except OCRError as e:
    ...     print(f"OCR failed: {e}")
"""


class ToolError(Exception):
    """Base exception for all tool-related errors.

    All custom tool exceptions inherit from this base class, allowing
    catch-all error handling when needed.

    Attributes
    ----------
    message : str
        Human-readable error description
    context : dict
        Additional context about the error (optional)

    Examples
    --------
    >>> raise ToolError("Tool initialization failed", context={"tool": "OCR"})
    """

    def __init__(self, message: str, context: dict | None = None) -> None:
        """Initialize ToolError with message and optional context.

        Parameters
        ----------
        message : str
            Human-readable error description
        context : dict | None, optional
            Additional error context (e.g., model name, input size)
        """
        super().__init__(message)
        self.message = message
        self.context = context or {}

    def __str__(self) -> str:
        """Return formatted error message with context."""
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} ({context_str})"
        return self.message


class OCRError(ToolError):
    """OCR text extraction failed.

    Raised when optical character recognition cannot extract text from an image.
    Common causes: invalid image format, corrupted data, unsupported language.

    Examples
    --------
    >>> raise OCRError("Image format not supported", context={"format": "HEIC"})
    """

    pass


class CaptionError(ToolError):
    """Image captioning failed.

    Raised when image caption generation fails. Common causes: invalid image,
    model loading error, out of memory.

    Examples
    --------
    >>> raise CaptionError("Model failed to load", context={"model": "blip2"})
    """

    pass


class LLMError(ToolError):
    """LLM prediction failed.

    Raised when language model prediction fails. Common causes: API timeout,
    rate limiting, invalid prompt, authentication failure.

    Examples
    --------
    >>> raise LLMError("API rate limit exceeded", context={"model": "gpt-4"})
    """

    pass


class VLMError(ToolError):
    """Vision-Language Model analysis failed.

    Raised when VLM image analysis fails. Common causes: invalid image format,
    API timeout, rate limiting, unsupported provider, authentication failure.

    Examples
    --------
    >>> raise VLMError(
    ...     "Image analysis failed", context={"provider": "openai", "model": "gpt-4o"}
    ... )
    """

    pass


class EmbeddingError(ToolError):
    """Text embedding generation failed.

    Raised when converting text to vector embeddings fails. Common causes:
    model loading error, text too long, out of memory.

    Examples
    --------
    >>> raise EmbeddingError(
    ...     "Text exceeds max length", context={"text_length": 10000, "max_length": 512}
    ... )
    """

    pass


class FAISSError(ToolError):
    """FAISS index search failed.

    Raised when vector similarity search fails. Common causes: index not found,
    corrupted index, dimension mismatch, invalid query vector.

    Examples
    --------
    >>> raise FAISSError(
    ...     "Index file not found", context={"index_path": "/data/risks.faiss"}
    ... )
    """

    pass


class FormatterError(ToolError):
    """Output formatting failed.

    Raised when generating formatted output (HTML, tables) fails. Common causes:
    invalid template, malformed data structure.

    Examples
    --------
    >>> raise FormatterError("Invalid HTML template", context={"template": "highlight"})
    """

    pass


class PersistenceError(ToolError):
    """File persistence operation failed.

    Raised when saving data to disk fails. Common causes: permission denied,
    disk full, invalid path.

    Examples
    --------
    >>> raise PersistenceError(
    ...     "Permission denied", context={"path": "/protected/output.csv"}
    ... )
    """

    pass


class ToolConfigurationError(ToolError):
    """Tool configuration or initialization failed.

    Raised when a tool cannot be constructed from configuration. Common causes:
    unknown tool name, missing dependencies, invalid settings.

    Examples
    --------
    >>> raise ToolConfigurationError(
    ...     "Unknown OCR tool",
    ...     context={
    ...         "ocr_tool": "invalid_tool",
    ...         "valid_options": ["tesseract", "paddleocr"],
    ...     },
    ... )
    """

    pass
