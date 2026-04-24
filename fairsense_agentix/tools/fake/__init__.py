"""Fake tool implementations for testing and development.

This package provides simple, deterministic fake implementations of all tool
protocols. These fakes are used for:
    - Development without expensive API calls or heavy dependencies
    - Testing workflows in isolation
    - Demonstrating system functionality (Phase 2-4)

All fake tools:
    - Return deterministic outputs (same input -> same output)
    - Track call counts and arguments for testing
    - Support custom return values for specific test scenarios
    - Satisfy their respective protocol interfaces

Implementation is split across focused modules:
    - :mod:`.image`  — FakeOCRTool, FakeCaptionTool
    - :mod:`.text`   — FakeLLMTool, FakeSummarizerTool
    - :mod:`.search` — FakeEmbedderTool, FakeFAISSIndexTool
    - :mod:`.output` — FakeFormatterTool, FakePersistenceTool

All public names are re-exported here so existing imports of the form
``from fairsense_agentix.tools.fake import ...`` continue to work.

Examples
--------
    >>> from fairsense_agentix.tools.fake import FakeOCRTool
    >>> ocr = FakeOCRTool()
    >>> text = ocr.extract(b"image_data")
    >>> print(ocr.call_count)  # Track calls
    1
"""

from fairsense_agentix.tools.fake.image import FakeCaptionTool, FakeOCRTool
from fairsense_agentix.tools.fake.output import FakeFormatterTool, FakePersistenceTool
from fairsense_agentix.tools.fake.search import FakeEmbedderTool, FakeFAISSIndexTool
from fairsense_agentix.tools.fake.text import FakeLLMTool, FakeSummarizerTool


__all__ = [
    "FakeOCRTool",
    "FakeCaptionTool",
    "FakeLLMTool",
    "FakeSummarizerTool",
    "FakeEmbedderTool",
    "FakeFAISSIndexTool",
    "FakeFormatterTool",
    "FakePersistenceTool",
]
