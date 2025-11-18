"""OCR (Optical Character Recognition) tools for text extraction from images.

This module provides OCR implementations with GPU-adaptive selection:
- TesseractOCR: CPU-friendly, 100+ languages, good accuracy (85-90%)
- PaddleOCR: GPU-accelerated, excellent accuracy (95%+), handles rotated text

The registry automatically selects the best tool based on GPU availability.
"""

from fairsense_agentix.tools.ocr.tesseract_tool import TesseractOCRTool


__all__ = ["TesseractOCRTool"]

# PaddleOCRTool exported conditionally (optional dependency)
try:
    from fairsense_agentix.tools.ocr.paddleocr_tool import PaddleOCRTool

    __all__.append("PaddleOCRTool")
except ImportError:
    # Silently ignore if PaddleOCR dependencies not installed
    # This is expected behavior - PaddleOCR is optional (GPU-only)
    # Registry will fallback to TesseractOCR (CPU) or FakeOCRTool
    pass
