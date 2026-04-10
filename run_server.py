#!/usr/bin/env python3
"""Simple script to run the FairSense AgentiX server with proper error handling."""

import logging
import sys


logger = logging.getLogger(__name__)

try:
    from fairsense_agentix.logging_config import ensure_root_logging
except ImportError:

    def ensure_root_logging(level: int = logging.INFO) -> None:
        """Fallback when package is not installed; same behavior as logging_config."""
        if not logging.root.handlers:
            logging.basicConfig(
                level=level,
                format="%(levelname)s:%(name)s:%(message)s",
            )


try:
    import uvicorn

    from fairsense_agentix.configs.settings import settings

    # Root handler + levels: logging_config runs on import of ensure_root_logging.
    ensure_root_logging(getattr(logging, settings.log_level))

    logger.info("=" * 70)
    logger.info("Starting FairSense AgentiX Server")
    logger.info("=" * 70)
    logger.info("Host: %s", settings.api_host)
    logger.info("Port: %s", settings.api_port)
    logger.info("Reload: %s", settings.api_reload)
    logger.info("LLM Provider: %s", settings.llm_provider)
    logger.info("=" * 70)

    # Run uvicorn
    uvicorn.run(
        "fairsense_agentix.service_api.server:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level="info",
    )

except KeyboardInterrupt:
    ensure_root_logging()
    logger.info("Server stopped by user")
    sys.exit(0)

except Exception:
    ensure_root_logging()
    logger.exception("Failed to start server")
    sys.exit(1)
