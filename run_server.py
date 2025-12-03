#!/usr/bin/env python3
"""Simple script to run the FairSense AgentiX server with proper error handling."""

import sys


try:
    import uvicorn

    from fairsense_agentix.configs.settings import settings

    print("=" * 70)
    print("Starting FairSense AgentiX Server")
    print("=" * 70)
    print(f"Host: {settings.api_host}")
    print(f"Port: {settings.api_port}")
    print(f"Reload: {settings.api_reload}")
    print(f"LLM Provider: {settings.llm_provider}")
    print("=" * 70)
    print()

    # Run uvicorn
    uvicorn.run(
        "fairsense_agentix.service_api.server:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level="info",
    )

except KeyboardInterrupt:
    print("\n\nServer stopped by user")
    sys.exit(0)

except Exception as e:
    print("\n\nERROR: Failed to start server")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    print("\nFull traceback:")
    import traceback

    traceback.print_exc()
    sys.exit(1)
