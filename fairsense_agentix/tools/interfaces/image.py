"""Image processing tool protocol interfaces."""

from typing import Protocol, runtime_checkable


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
