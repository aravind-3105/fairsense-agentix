"""Server launcher module for FairSense-AgentiX.

This module provides a programmatic way to start both the FastAPI backend
and React frontend with a single function call.

Performance Optimization
------------------------
This module automatically disables eager loading when imported to prevent
double model loading:
1. Parent process: NO eager loading (fast import, ~1-2s)
2. Backend subprocess: YES eager loading (loads models once, ~30-45s)

This saves 20-30s of unnecessary startup time while keeping all benefits
of eager loading in the backend server.

Examples
--------
Simple usage (starts on default ports):

    >>> from fairsense_agentix import server
    >>> server.start()

Custom configuration:

    >>> server.start(port=9000, ui_port=3000, verbose=False)

Development mode with auto-reload:

    >>> server.start(reload=True)
"""

import os


# CRITICAL: Disable eager loading when importing server module
# The parent process doesn't need models loaded - only the backend subprocess does
# This prevents double loading and saves 20-30s startup time
os.environ.setdefault("FAIRSENSE_DISABLE_EAGER_LOADING", "true")

from fairsense_agentix.server.launcher import start


__all__ = ["start"]
