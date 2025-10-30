"""Tools module for FairSense-AgentiX.

This module provides the tool abstraction layer that enables:
    - Swappable tool implementations (fake vs real)
    - Protocol-based contracts for type safety
    - Configuration-driven tool selection via registry
    - Dependency injection for graphs

Key Components:
    - Protocols: Abstract interfaces in interfaces.py
    - Implementations: Fake tools in fake.py (real tools in Phase 5)
    - Registry: Factory for constructing and wiring tools
    - Exceptions: Typed exceptions for clear error handling

Examples
--------
    >>> from fairsense_agentix.tools import get_tool_registry
    >>> registry = get_tool_registry()
    >>> text = registry.ocr.extract(b"image_bytes")
    >>> analysis = registry.llm.predict("Analyze this text...")
"""

# Protocols (contracts)
# Exceptions
from fairsense_agentix.tools.exceptions import (
    CaptionError,
    EmbeddingError,
    FAISSError,
    FormatterError,
    LLMError,
    OCRError,
    PersistenceError,
    ToolConfigurationError,
    ToolError,
)

# Fake implementations (for testing and Phase 4)
from fairsense_agentix.tools.fake import (
    FakeCaptionTool,
    FakeEmbedderTool,
    FakeFAISSIndexTool,
    FakeFormatterTool,
    FakeLLMTool,
    FakeOCRTool,
    FakePersistenceTool,
    FakeSummarizerTool,
)
from fairsense_agentix.tools.interfaces import (
    CaptionTool,
    EmbedderTool,
    FAISSIndexTool,
    FormatterTool,
    LLMTool,
    OCRTool,
    PersistenceTool,
    SummarizerTool,
)

# Registry (factory)
from fairsense_agentix.tools.registry import (
    ToolRegistry,
    create_tool_registry,
    get_tool_registry,
    reset_tool_registry,
)


__all__ = [
    # Protocols
    "OCRTool",
    "CaptionTool",
    "LLMTool",
    "SummarizerTool",
    "EmbedderTool",
    "FAISSIndexTool",
    "FormatterTool",
    "PersistenceTool",
    # Registry
    "ToolRegistry",
    "create_tool_registry",
    "get_tool_registry",
    "reset_tool_registry",
    # Exceptions
    "ToolError",
    "OCRError",
    "CaptionError",
    "LLMError",
    "EmbeddingError",
    "FAISSError",
    "FormatterError",
    "PersistenceError",
    "ToolConfigurationError",
    # Fake implementations
    "FakeOCRTool",
    "FakeCaptionTool",
    "FakeLLMTool",
    "FakeSummarizerTool",
    "FakeEmbedderTool",
    "FakeFAISSIndexTool",
    "FakeFormatterTool",
    "FakePersistenceTool",
]
