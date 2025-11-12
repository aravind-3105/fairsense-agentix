"""Tesseract OCR tool for CPU-based text extraction from images.

This module implements OCR using Tesseract, Google's open-source OCR engine.
Tesseract is optimized for CPU execution and supports 100+ languages.

Design Choices
--------------
1. **Why Tesseract**: CPU-friendly (50-100ms), no GPU needed, 100+ languages
2. **Why CPU-only**: Tesseract doesn't benefit from GPU, optimized for CPU
3. **Why pytesseract**: Official Python wrapper, stable, well-maintained
4. **Confidence filtering**: Only return text with confidence >= threshold

What This Enables
-----------------
- Fast OCR on CPU-only machines (laptops, containers without GPU)
- Multilingual support (100+ languages)
- Reliable fallback when GPU unavailable
- No large model downloads (uses system tesseract binary)
"""

import io
import logging

from PIL import Image

from fairsense_agentix.tools.exceptions import OCRError


logger = logging.getLogger(__name__)


class TesseractOCRTool:
    """Tesseract OCR tool for CPU-based text extraction.

    Uses Google's Tesseract OCR engine via pytesseract wrapper. Optimized for
    CPU execution with no GPU requirement. Supports 100+ languages.

    Performance
    -----------
    - CPU: 50-100ms per image
    - No GPU benefit (CPU-optimized)

    Accuracy
    --------
    - Clean text: 85-90%
    - Challenging images (blur, rotation): 60-80%

    Examples
    --------
    >>> ocr = TesseractOCRTool()
    >>> text = ocr.extract(image_bytes, language="eng", confidence_threshold=0.5)
    >>> print(text)
    'Job Opening: Software Engineer'

    Notes
    -----
    Requires system tesseract binary installed:
    - Ubuntu/Debian: sudo apt-get install tesseract-ocr
    - macOS: brew install tesseract
    - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
    """

    def __init__(self) -> None:
        """Initialize Tesseract OCR tool.

        Design Choice: Validate Tesseract installation at init time (fail fast)
        What This Enables: Clear error message if tesseract not installed
        Why This Way: Better UX than cryptic error during first extract() call

        Raises
        ------
        OCRError
            If tesseract binary not found on system
        """
        self._validate_tesseract_installed()
        logger.info("TesseractOCRTool initialized (CPU mode)")

    def _validate_tesseract_installed(self) -> None:
        """Validate that tesseract binary is installed.

        Design Choice: Separate validation method for clarity and testability
        What This Enables: Easy mocking in tests, clear separation of concerns
        Why This Way: Single responsibility - validation logic isolated

        Raises
        ------
        OCRError
            If tesseract binary not found
        """
        try:
            import pytesseract  # noqa: PLC0415

            pytesseract.get_tesseract_version()
        except ImportError as e:
            msg = "pytesseract not installed. Install with: pip install pytesseract"
            raise OCRError(msg) from e
        except Exception as e:
            msg = (
                "Tesseract binary not found. Install tesseract-ocr:\n"
                "  Ubuntu/Debian: sudo apt-get install tesseract-ocr\n"
                "  macOS: brew install tesseract\n"
                "  Windows: https://github.com/UB-Mannheim/tesseract/wiki"
            )
            raise OCRError(msg, context={"error": str(e)}) from e

    def extract(
        self,
        image_bytes: bytes,
        language: str = "eng",
        confidence_threshold: float = 0.5,
    ) -> str:
        """Extract text from image using Tesseract OCR.

        Design Choices:
        - **image_to_data() over image_to_string()**: Get confidence scores
          for filtering
        - **Confidence filtering**: Only include text with confidence >= threshold
        - **Space joining**: Preserve word boundaries in extracted text
        - **Error context**: Include language and image size in error messages

        What This Enables:
        - Quality filtering: Remove low-confidence OCR artifacts
        - Multi-language: Support any language Tesseract has trained data for
        - Robust error handling: Clear errors with actionable context

        Why This Way:
        - image_to_data() returns structured data (text + confidence + bbox)
        - Confidence filtering reduces noise from compression artifacts
        - Context in errors helps debugging (was it language issue? corrupt image?)

        Parameters
        ----------
        image_bytes : bytes
            Raw image data (PNG, JPEG, etc.)
        language : str, optional
            Tesseract language code (e.g., 'eng', 'fra', 'spa'), by default "eng"
        confidence_threshold : float, optional
            Minimum confidence (0.0-1.0) for including detected text, by default 0.5

        Returns
        -------
        str
            Extracted text joined with spaces

        Raises
        ------
        OCRError
            If image invalid, unsupported format, or OCR fails

        Examples
        --------
        >>> ocr = TesseractOCRTool()
        >>> text = ocr.extract(image_bytes)
        >>> print(text)
        'Looking for a young energetic salesman'

        >>> # Multi-language
        >>> text = ocr.extract(french_image, language="fra")

        >>> # High quality only
        >>> text = ocr.extract(image_bytes, confidence_threshold=0.8)
        """
        try:
            import pytesseract  # noqa: PLC0415

            # Convert bytes → PIL Image
            # Design Choice: Use PIL for compatibility with pytesseract
            # What This Enables: Handles all image formats PIL supports
            image = Image.open(io.BytesIO(image_bytes))

            # Extract text with confidence scores
            # Design Choice: Use image_to_data() for structured output
            # What This Enables: Access to confidence scores for filtering
            # Why This Way: image_to_string() doesn't provide confidence
            data = pytesseract.image_to_data(
                image,
                lang=language,
                output_type=pytesseract.Output.DICT,
            )

            # Filter by confidence and collect text
            # Design Choice: List comprehension for performance
            # What This Enables: Efficient filtering + joining in single pass
            texts = [
                data["text"][i]
                for i in range(len(data["text"]))
                if data["conf"][i] != -1  # -1 = no text detected
                and data["conf"][i]
                >= confidence_threshold * 100  # Convert to 0-100 scale
                and data["text"][i].strip()  # Skip empty strings
            ]

            result = " ".join(texts)
            logger.debug(
                f"Tesseract extracted {len(texts)} text segments "
                f"({len(result)} chars total, language={language})"
            )
            return result

        except OSError as e:
            # Image format issues
            msg = "Invalid or corrupted image"
            raise OCRError(
                msg,
                context={
                    "language": language,
                    "image_size_bytes": len(image_bytes),
                    "error": str(e),
                },
            ) from e
        except Exception as e:
            # Catch-all for unexpected errors
            msg = "OCR extraction failed"
            raise OCRError(
                msg,
                context={
                    "language": language,
                    "image_size_bytes": len(image_bytes),
                    "error": str(e),
                },
            ) from e
