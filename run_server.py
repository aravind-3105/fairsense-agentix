#!/usr/bin/env python3
"""Simple script to run the FairSense AgentiX server with proper error handling."""

import logging
import sys


logger = logging.getLogger(__name__)


try:
    import uvicorn

    from fairsense_agentix import logging_config  # noqa: F401 (side effects)
    from fairsense_agentix.configs.settings import settings

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
    logger.info("Server stopped by user")
    sys.exit(0)

except Exception:
    logger.exception("Failed to start server")
    sys.exit(1)
