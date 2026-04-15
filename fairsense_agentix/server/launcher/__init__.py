"""Server launcher package for FairSense-AgentiX.

Manages the full lifecycle of backend (FastAPI) and frontend (React/Vite)
server processes with health checking, graceful shutdown, and cross-platform
support.

Implementation is split across focused modules:
    - :mod:`.messages`  — User-facing log output (banners, troubleshooting)
    - :mod:`.health`    — HTTP health-check polling for backend and frontend
    - :mod:`.processes` — Subprocess management (start, kill)
    - :mod:`.core`      — ServerLauncher orchestrator and start() entry point

All public names are re-exported here so existing imports of the form
``from fairsense_agentix.server.launcher import ...`` continue to work.

Examples
--------
    >>> from fairsense_agentix.server.launcher import start
    >>> start()  # Launches both servers and blocks until Ctrl+C
"""

from fairsense_agentix.server.launcher.core import ServerLauncher, start


__all__ = ["ServerLauncher", "start"]
