"""Troubleshooting log messages for server launcher failures."""

import logging


# Keep log records under the same logger name as ServerLauncher for filtering.
logger = logging.getLogger("fairsense_agentix.server.launcher")


def log_backend_troubleshooting(backend_port: int) -> None:
    """Log backend troubleshooting instructions."""
    logger.error(
        "❌ Backend failed to start on port %s",
        backend_port,
    )
    logger.info("💡 Troubleshooting:")
    logger.info("   • Check if port %s is already in use", backend_port)
    logger.info("   • Run: lsof -i :%s  (macOS/Linux)", backend_port)
    logger.info(
        "   • Run: netstat -ano | findstr :%s  (Windows)",
        backend_port,
    )
    logger.info(
        "   • Check .env file for FAIRSENSE_LLM_API_KEY and other settings",
    )


def log_frontend_troubleshooting(frontend_port: int) -> None:
    """Log frontend troubleshooting instructions."""
    logger.error(
        "❌ Frontend failed to start on port %s",
        frontend_port,
    )
    logger.info("💡 Troubleshooting:")
    logger.info(
        "   • Check if port %s is already in use",
        frontend_port,
    )
    logger.info(
        "   • Ensure npm dependencies are installed: cd ui && npm install",
    )
    logger.info(
        "   • Check ui/ directory exists and contains package.json",
    )


def log_nodejs_install_instructions() -> None:
    """Log Node.js installation instructions."""
    logger.error("❌ npm not found")
    logger.info("💡 Install Node.js:")
    logger.info("   • macOS: brew install node")
    logger.info("   • Windows: https://nodejs.org/")
    logger.info("   • Linux: sudo apt install nodejs npm")
