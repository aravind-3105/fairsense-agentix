"""HTTP health checks for backend and frontend processes."""

import logging
import subprocess
import time
from typing import Optional

import requests


logger = logging.getLogger("fairsense_agentix.server.launcher")


def _debug(verbose: bool, msg: str, *args: object) -> None:
    if verbose:
        logger.debug(msg, *args)


def wait_for_backend(  # noqa: PLR0912
    backend_proc: Optional[subprocess.Popen],
    backend_port: int,
    *,
    timeout: int = 120,
    verbose: bool = False,
) -> bool:
    """Poll /v1/health until success or timeout."""
    urls = [
        f"http://localhost:{backend_port}/v1/health",
        f"http://127.0.0.1:{backend_port}/v1/health",
    ]
    start_time = time.time()

    _debug(verbose, "Starting health checks with %ss timeout", timeout)
    _debug(verbose, "URLs to try: %s", urls)

    attempt = 0
    while time.time() - start_time < timeout:
        attempt += 1

        if backend_proc is not None:
            poll_status = backend_proc.poll()
            if poll_status is not None:
                logger.warning(
                    "Backend process exited with code %s (health check aborted)",
                    poll_status,
                )
                return False

            if attempt % 5 == 1:
                _debug(
                    verbose,
                    "Attempt %s - backend PID %s running",
                    attempt,
                    backend_proc.pid,
                )

        for url in urls:
            try:
                if attempt <= 3:
                    _debug(verbose, "Trying: %s", url)
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    _debug(
                        verbose,
                        "Health check success: %s %s",
                        response.status_code,
                        url,
                    )
                    return True
            except requests.exceptions.ConnectionError:
                if attempt <= 3:
                    _debug(verbose, "Server not ready")
            except requests.exceptions.Timeout:
                if attempt <= 3:
                    _debug(verbose, "Request timeout (server slow to respond)")
            except requests.exceptions.RequestException as e:
                if attempt <= 3:
                    _debug(
                        verbose,
                        "Request error: %s: %s",
                        type(e).__name__,
                        e,
                    )

        time.sleep(1)

    _debug(verbose, "Health check TIMEOUT after %ss", timeout)
    if backend_proc is not None and hasattr(backend_proc, "poll"):
        _debug(
            verbose,
            "Backend process status: %s",
            backend_proc.poll(),
        )
    return False


def wait_for_frontend(frontend_port: int, *, timeout: int = 30) -> bool:
    """Poll frontend root URL until HTTP 200 or timeout."""
    urls = [
        f"http://localhost:{frontend_port}",
        f"http://127.0.0.1:{frontend_port}",
    ]
    start_time = time.time()

    while time.time() - start_time < timeout:
        for url in urls:
            try:
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                pass
        time.sleep(1)

    return False
