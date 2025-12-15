"""Example: Launch FairSense-AgentiX server programmatically.

This script demonstrates the simplest way to start FairSense-AgentiX
with both backend and frontend servers.

Usage
-----
From project root:

    python examples/launch_server.py

Or with Python API:

    from fairsense_agentix import server
    server.start()

This will:
- Start FastAPI backend on http://localhost:8000
- Start React UI on http://localhost:5173
- Auto-open your browser
- Block until Ctrl+C pressed

Note: The server module automatically disables eager loading in the parent
process to prevent double model loading. No manual configuration needed!
"""

from fairsense_agentix import server


if __name__ == "__main__":
    # Simple usage - all defaults
    server.start()

    # Advanced usage examples (commented out):

    # Custom ports
    # server.start(port=9000, ui_port=3000)

    # Clean output (status only, no logs)
    # server.start(verbose=False)

    # Development mode (auto-reload on code changes)
    # server.start(reload=True)

    # Headless (no browser auto-open)
    # server.start(open_browser=False)
