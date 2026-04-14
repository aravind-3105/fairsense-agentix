"""Tool protocol interfaces for FairSense-AgentiX.

This module defines Protocol classes (abstract interfaces) that all tool
implementations must satisfy. Protocols enable:
    - Structural subtyping (duck typing with type checking)
    - Swappable implementations (fake, real, mock)
    - Clear contracts for tool behavior
    - Type safety without inheritance requirements

Protocols use Python's typing.Protocol with @runtime_checkable decorator,
allowing isinstance() checks at runtime.

Implementation is split across focused modules:
    - :mod:`.image`  — OCRTool, CaptionTool
    - :mod:`.text`   — LLMTool, SummarizerTool, VLMTool
    - :mod:`.search` — EmbedderTool, FAISSIndexTool
    - :mod:`.output` — FormatterTool, PersistenceTool

All public names are re-exported here so existing imports of the form
``from fairsense_agentix.tools.interfaces import ...`` continue to work.

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

from fairsense_agentix.tools.interfaces.image import CaptionTool, OCRTool
from fairsense_agentix.tools.interfaces.output import FormatterTool, PersistenceTool
from fairsense_agentix.tools.interfaces.search import EmbedderTool, FAISSIndexTool
from fairsense_agentix.tools.interfaces.text import LLMTool, SummarizerTool, VLMTool


__all__ = [
    "OCRTool",
    "CaptionTool",
    "LLMTool",
    "SummarizerTool",
    "VLMTool",
    "EmbedderTool",
    "FAISSIndexTool",
    "FormatterTool",
    "PersistenceTool",
]
