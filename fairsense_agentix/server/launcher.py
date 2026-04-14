"""Server launcher: orchestrates backend and frontend processes.

Process startup, health polling, troubleshooting logs, and port cleanup live in
companion modules under :mod:`fairsense_agentix.server`.
"""

import logging
import signal
import subprocess
import sys
import time
import webbrowser
from typing import Optional

from fairsense_agentix.logging_config import ensure_root_logging
from fairsense_agentix.server import (
    launcher_health,
    launcher_ports,
    launcher_processes,
    launcher_troubleshooting,
)


logger = logging.getLogger(__name__)


class ServerLauncher:
    """Manages backend and frontend server processes.

    Parameters
    ----------
    backend_port : int, default=8000
        Port for FastAPI backend
    frontend_port : int, default=5173
        Port for React UI (Vite default)
    open_browser : bool, default=True
        Auto-open browser when ready
    verbose : bool, default=True
        Stream backend/frontend logs to stdout (False = status only)
    reload : bool, default=False
        Enable uvicorn auto-reload (for prompt/config customization)

    Examples
    --------
    >>> launcher = ServerLauncher()
    >>> launcher.start()
    >>> launcher.wait()  # Blocks until Ctrl+C
    """

    def __init__(
        self,
        backend_port: int = 8000,
        frontend_port: int = 5173,
        open_browser: bool = True,
        verbose: bool = True,
        reload: bool = False,
    ) -> None:
        """Initialize launcher with configuration."""
        self.backend_port = backend_port
        self.frontend_port = frontend_port
        self.open_browser = open_browser
        self.verbose = verbose
        self.reload = reload

        logger.setLevel(logging.DEBUG if verbose else logging.INFO)

        self.backend_proc: Optional[subprocess.Popen] = None
        self.frontend_proc: Optional[subprocess.Popen] = None

        self._setup_signal_handlers()

    def _log_debug(self, msg: str, *args: object) -> None:
        launcher_ports.log_debug(self.verbose, msg, *args)

    def _setup_signal_handlers(self) -> None:
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, _signum: int, _frame: object) -> None:
        logger.info("🛑 Shutting down servers...")
        self.stop()
        sys.exit(0)

    def start(self) -> tuple[subprocess.Popen, subprocess.Popen]:
        """Start backend and frontend; block until both pass health checks."""
        self._print_banner()
        self._start_backend_with_health_check()
        self._start_frontend_with_health_check()
        self._open_browser_if_enabled()
        self._print_ready_message()

        assert self.backend_proc is not None, "Backend process must be set"
        assert self.frontend_proc is not None, "Frontend process must be set"
        return self.backend_proc, self.frontend_proc

    def _start_backend_with_health_check(self) -> None:
        logger.info("▶️  Starting backend on port %s...", self.backend_port)
        try:
            self.backend_proc = launcher_processes.start_backend_process(
                self.backend_port,
                self.reload,
                self.verbose,
            )
        except Exception as e:
            logger.error("❌ Failed to start backend: %s", e)
            self.stop()
            sys.exit(1)

        logger.info(
            "⏳ Waiting for backend to be ready (preloading models, 1-2 minutes)...",
        )
        logger.info("   Initial startup delay before health checks: 10 seconds...")
        time.sleep(10)

        if not launcher_health.wait_for_backend(
            self.backend_proc,
            self.backend_port,
            timeout=120,
            verbose=self.verbose,
        ):
            launcher_troubleshooting.log_backend_troubleshooting(self.backend_port)
            self.stop()
            sys.exit(1)
        logger.info("✅ Backend ready")

    def _start_frontend_with_health_check(self) -> None:
        logger.info("▶️  Starting frontend on port %s...", self.frontend_port)
        try:
            self.frontend_proc = launcher_processes.start_frontend_process(
                self.backend_port,
                self.frontend_port,
                self.verbose,
            )
        except FileNotFoundError:
            launcher_troubleshooting.log_nodejs_install_instructions()
            self.stop()
            sys.exit(1)
        except Exception as e:
            logger.error("❌ Failed to start frontend: %s", e)
            self.stop()
            sys.exit(1)

        logger.info("⏳ Waiting for frontend to be ready...")
        if not launcher_health.wait_for_frontend(
            self.frontend_port,
            timeout=30,
        ):
            launcher_troubleshooting.log_frontend_troubleshooting(self.frontend_port)
            self.stop()
            sys.exit(1)
        logger.info("✅ Frontend ready")

    def _open_browser_if_enabled(self) -> None:
        if self.open_browser:
            logger.info(
                "🌐 Opening browser at http://localhost:%s",
                self.frontend_port,
            )
            time.sleep(1)
            webbrowser.open(f"http://localhost:{self.frontend_port}")

    def stop(self) -> None:
        """Terminate subprocesses and best-effort free listening ports."""
        if self.frontend_proc:
            try:
                if self.frontend_proc.poll() is None:
                    self.frontend_proc.terminate()
                    try:
                        self.frontend_proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        self.frontend_proc.kill()
                        self.frontend_proc.wait(timeout=1)
                        launcher_ports.kill_port_listeners(
                            self.frontend_port,
                            verbose=self.verbose,
                        )
            except Exception:
                pass

        if self.backend_proc:
            try:
                if self.backend_proc.poll() is None:
                    self.backend_proc.terminate()
                    try:
                        self.backend_proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        self.backend_proc.kill()
                        self.backend_proc.wait(timeout=1)
                        launcher_ports.kill_port_listeners(
                            self.backend_port,
                            verbose=self.verbose,
                        )
            except Exception:
                pass

        logger.info("✅ Servers stopped")

    def wait(self) -> None:
        """Block until backend exits, user interrupt, or frontend alone exits."""
        try:
            if self.backend_proc:
                self.backend_proc.wait()
                logger.info("🛑 Backend process exited")

                if self.frontend_proc and self.frontend_proc.poll() is None:
                    logger.info("🛑 Shutting down frontend...")
                    self.stop()
                return

            if self.frontend_proc:
                self.frontend_proc.wait()
        except KeyboardInterrupt:
            logger.info("🛑 Shutting down servers...")
            self.stop()

    def _print_banner(self) -> None:
        logger.info("=" * 70)
        logger.info("🚀 FairSense-AgentiX Server Launcher")
        logger.info("=" * 70)

    def _print_ready_message(self) -> None:
        logger.info("=" * 70)
        logger.info("✅ FairSense-AgentiX is running!")
        logger.info("=" * 70)
        logger.info("Backend:  http://localhost:%s", self.backend_port)
        logger.info("Frontend: http://localhost:%s", self.frontend_port)
        logger.info("API Docs: http://localhost:%s/docs", self.backend_port)
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 70)


def start(
    port: int = 8000,
    ui_port: int = 5173,
    open_browser: bool = True,
    verbose: bool = True,
    reload: bool = False,
) -> None:
    """Start FairSense-AgentiX with backend and frontend servers."""
    ensure_root_logging(logging.DEBUG if verbose else logging.INFO)

    launcher = ServerLauncher(
        backend_port=port,
        frontend_port=ui_port,
        open_browser=open_browser,
        verbose=verbose,
        reload=reload,
    )

    launcher.start()
    launcher.wait()
