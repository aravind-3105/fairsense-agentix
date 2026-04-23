"""Port cleanup helpers for server launcher."""

import logging
import platform
import subprocess


logger = logging.getLogger("fairsense_agentix.server.launcher")


def log_debug(verbose: bool, msg: str, *args: object) -> None:
    """Emit DEBUG when verbose launcher mode is on."""
    if verbose:
        logger.debug(msg, *args)


def kill_port_listeners(port: int, *, verbose: bool) -> None:
    """Kill any process listening on the specified port (best effort)."""
    try:
        system = platform.system()

        if system == "Windows":
            subprocess.run(
                (
                    f'for /f "tokens=5" %a in '
                    f"('netstat -aon ^| findstr :{port}') "
                    f"do taskkill /F /PID %a"
                ),
                check=False,
                shell=True,
                capture_output=True,
            )
        else:
            subprocess.run(
                f"lsof -ti :{port} | xargs kill -9",
                check=False,
                shell=True,
                capture_output=True,
            )

        log_debug(verbose, "Cleaned up port %s", port)
    except Exception as e:
        logger.warning("Port cleanup warning: %s", e)
