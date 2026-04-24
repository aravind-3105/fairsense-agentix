"""Fake image tool implementations for testing."""

from typing import Any


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

        caption = f"A photo or illustration ({len(image_bytes)} bytes)"
        return caption[:max_length]
