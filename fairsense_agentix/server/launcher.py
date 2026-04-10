"""Main server launcher implementation.

This module provides the ServerLauncher class that manages backend (FastAPI)
and frontend (React/Vite) server processes with health checking, graceful
shutdown, and cross-platform support.
"""

import logging
import os
import platform
import signal
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import Optional

import requests

from fairsense_agentix import logging_config  # noqa: F401 (imported for side effects)


logger = logging.getLogger(__name__)


class ServerLauncher:
    """Manages backend and frontend server processes.

    This class handles the full lifecycle of both servers:
    1. Start backend (FastAPI + uvicorn)
    2. Health check backend (/v1/health endpoint)
    3. Start frontend (React + Vite dev server)
    4. Health check frontend (HTTP GET /)
    5. Open browser (optional)
    6. Wait for user interrupt (Ctrl+C)
    7. Graceful shutdown

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

        # So logger.debug() from _log_debug() is emitted when verbose=True
        # (parent logger is INFO)
        logger.setLevel(logging.DEBUG if verbose else logging.INFO)

        self.backend_proc: Optional[subprocess.Popen] = None
        self.frontend_proc: Optional[subprocess.Popen] = None

        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()

    def _log_debug(self, msg: str, *args: object) -> None:
        """Log at DEBUG only when verbose=True (keeps quiet mode clean)."""
        if self.verbose:
            logger.debug(msg, *args)

    def _setup_signal_handlers(self) -> None:
        """Register signal handlers for graceful shutdown on Ctrl+C."""
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, _signum: int, _frame: object) -> None:
        """Handle shutdown signals (SIGINT/SIGTERM)."""
        logger.info("🛑 Shutting down servers...")
        self.stop()
        sys.exit(0)

    def start(self) -> tuple[subprocess.Popen, subprocess.Popen]:
        """Start both backend and frontend servers.

        Returns
        -------
        tuple[subprocess.Popen, subprocess.Popen]
            Backend and frontend process handles

        Raises
        ------
        RuntimeError
            If either server fails to start
        FileNotFoundError
            If npm or UI directory not found
        """
        self._print_banner()
        self._start_backend_with_health_check()
        self._start_frontend_with_health_check()
        self._open_browser_if_enabled()
        self._print_ready_message()

        # Assert processes are set (they will be if we reach here without sys.exit)
        assert self.backend_proc is not None, "Backend process must be set"
        assert self.frontend_proc is not None, "Frontend process must be set"
        return self.backend_proc, self.frontend_proc

    def _start_backend_with_health_check(self) -> None:
        """Start backend and wait for it to be ready."""
        logger.info("▶️  Starting backend on port %s...", self.backend_port)
        try:
            self.backend_proc = self._start_backend()
        except Exception as e:
            logger.error("❌ Failed to start backend: %s", e)
            self.stop()
            sys.exit(1)

        logger.info(
            "⏳ Waiting for backend to be ready (preloading models, 1-2 minutes)...",
        )
        logger.info("   Initial startup delay before health checks: 10 seconds...")
        time.sleep(10)

        if not self._wait_for_backend():
            self._print_backend_troubleshooting()
            self.stop()
            sys.exit(1)
        logger.info("✅ Backend ready")

    def _start_frontend_with_health_check(self) -> None:
        """Start frontend and wait for it to be ready."""
        logger.info("▶️  Starting frontend on port %s...", self.frontend_port)
        try:
            self.frontend_proc = self._start_frontend()
        except FileNotFoundError:
            self._print_nodejs_install_instructions()
            self.stop()
            sys.exit(1)
        except Exception as e:
            logger.error("❌ Failed to start frontend: %s", e)
            self.stop()
            sys.exit(1)

        logger.info("⏳ Waiting for frontend to be ready...")
        if not self._wait_for_frontend():
            self._print_frontend_troubleshooting()
            self.stop()
            sys.exit(1)
        logger.info("✅ Frontend ready")

    def _open_browser_if_enabled(self) -> None:
        """Open browser if configured to do so."""
        if self.open_browser:
            logger.info(
                "🌐 Opening browser at http://localhost:%s",
                self.frontend_port,
            )
            time.sleep(1)
            webbrowser.open(f"http://localhost:{self.frontend_port}")

    def _print_backend_troubleshooting(self) -> None:
        """Print backend troubleshooting instructions."""
        logger.error(
            "❌ Backend failed to start on port %s",
            self.backend_port,
        )
        logger.info("💡 Troubleshooting:")
        logger.info("   • Check if port %s is already in use", self.backend_port)
        logger.info("   • Run: lsof -i :%s  (macOS/Linux)", self.backend_port)
        logger.info(
            "   • Run: netstat -ano | findstr :%s  (Windows)",
            self.backend_port,
        )
        logger.info(
            "   • Check .env file for FAIRSENSE_LLM_API_KEY and other settings",
        )

    def _print_frontend_troubleshooting(self) -> None:
        """Print frontend troubleshooting instructions."""
        logger.error(
            "❌ Frontend failed to start on port %s",
            self.frontend_port,
        )
        logger.info("💡 Troubleshooting:")
        logger.info(
            "   • Check if port %s is already in use",
            self.frontend_port,
        )
        logger.info(
            "   • Ensure npm dependencies are installed: cd ui && npm install",
        )
        logger.info(
            "   • Check ui/ directory exists and contains package.json",
        )

    def _print_nodejs_install_instructions(self) -> None:
        """Print Node.js installation instructions."""
        logger.error("❌ npm not found")
        logger.info("💡 Install Node.js:")
        logger.info("   • macOS: brew install node")
        logger.info("   • Windows: https://nodejs.org/")
        logger.info("   • Linux: sudo apt install nodejs npm")

    def _start_backend(self) -> subprocess.Popen:
        """Start FastAPI backend via uvicorn.

        Uses run_server.py script which launches uvicorn with proper settings.

        Returns
        -------
        subprocess.Popen
            Backend process handle
        """
        # Prepare environment variables
        env = os.environ.copy()
        env["FAIRSENSE_API_PORT"] = str(self.backend_port)
        env["FAIRSENSE_API_RELOAD"] = "true" if self.reload else "false"

        # CRITICAL: Re-enable eager loading for backend subprocess
        # The parent process has FAIRSENSE_DISABLE_EAGER_LOADING=true
        # (set in server/__init__.py) but backend MUST load models
        if "FAIRSENSE_DISABLE_EAGER_LOADING" in env:
            del env["FAIRSENSE_DISABLE_EAGER_LOADING"]
            self._log_debug("Re-enabled eager loading for backend subprocess")

        # Locate run_server.py
        project_root = Path(__file__).parent.parent.parent
        script = project_root / "run_server.py"

        self._log_debug("Backend script path: %s", script)
        self._log_debug("Script exists: %s", script.exists())
        self._log_debug("Project root: %s", project_root)
        self._log_debug("Python executable: %s", sys.executable)
        self._log_debug("Backend port: %s", self.backend_port)

        if not script.exists():
            raise FileNotFoundError(
                f"Backend launcher script not found: {script}\n"
                f"Ensure you're running from the project root directory.",
            )

        # Start backend process
        self._log_debug("Starting backend subprocess...")
        if self.verbose:
            # Stream logs to stdout
            proc = subprocess.Popen(
                [sys.executable, str(script)],
                env=env,
                cwd=str(project_root),
            )
        else:
            # Suppress logs (status messages only)
            proc = subprocess.Popen(
                [sys.executable, str(script)],
                env=env,
                cwd=str(project_root),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        self._log_debug("Backend process started with PID: %s", proc.pid)
        self._log_debug("Process poll status: %s", proc.poll())
        return proc

    def _start_frontend(self) -> subprocess.Popen:
        """Start Vite dev server for React UI.

        Returns
        -------
        subprocess.Popen
            Frontend process handle

        Raises
        ------
        FileNotFoundError
            If npm not found or UI directory missing
        """
        # Prepare environment variables
        env = os.environ.copy()
        env["VITE_API_BASE"] = f"http://localhost:{self.backend_port}"

        # Locate UI directory
        project_root = Path(__file__).parent.parent.parent
        ui_dir = project_root / "ui"

        if not ui_dir.exists():
            raise FileNotFoundError(
                f"UI directory not found: {ui_dir}\n"
                f"Ensure you cloned the repository completely.",
            )

        # Check if node_modules exists
        node_modules = ui_dir / "node_modules"
        if not node_modules.exists():
            logger.info("⚠️  Node modules not found. Installing dependencies...")
            logger.info(
                "   This is a one-time setup (may take 1-2 minutes)...",
            )
            install_proc = subprocess.run(
                ["npm", "install"],
                check=False,
                cwd=str(ui_dir),
                capture_output=not self.verbose,
            )
            if install_proc.returncode != 0:
                raise RuntimeError(
                    "npm install failed. Check your internet connection and try again.",
                )

        # Start frontend with explicit port
        # (--port flag forces Vite to use it or fail)
        if self.verbose:
            # Stream logs to stdout
            proc = subprocess.Popen(
                [
                    "npm",
                    "run",
                    "dev",
                    "--",
                    "--port",
                    str(self.frontend_port),
                    "--strictPort",
                ],
                cwd=str(ui_dir),
                env=env,
            )
        else:
            # Suppress logs (status messages only)
            proc = subprocess.Popen(
                [
                    "npm",
                    "run",
                    "dev",
                    "--",
                    "--port",
                    str(self.frontend_port),
                    "--strictPort",
                ],
                cwd=str(ui_dir),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        self._log_debug("Frontend process started with PID: %s", proc.pid)
        return proc

    def _wait_for_backend(self, timeout: int = 120) -> bool:  # noqa: PLR0912
        """Wait for backend to respond to health checks.

        Tries both localhost and 127.0.0.1 for maximum compatibility
        (backend binds to 0.0.0.0 which may not respond to "localhost" on some systems).

        Parameters
        ----------
        timeout : int, default=90
            Maximum seconds to wait (backend startup can take 30-45s for model loading
            plus 5-10s for uvicorn to start listening)

        Returns
        -------
        bool
            True if backend is ready, False if timeout
        """
        # Try both localhost and 127.0.0.1 for compatibility
        urls = [
            f"http://localhost:{self.backend_port}/v1/health",
            f"http://127.0.0.1:{self.backend_port}/v1/health",
        ]
        start_time = time.time()

        self._log_debug("Starting health checks with %ss timeout", timeout)
        self._log_debug("URLs to try: %s", urls)

        attempt = 0
        while time.time() - start_time < timeout:
            attempt += 1

            # Check if backend process is still alive
            if self.backend_proc:
                poll_status = self.backend_proc.poll()
                if poll_status is not None:
                    logger.warning(
                        "Backend process exited with code %s (health check aborted)",
                        poll_status,
                    )
                    return False

                if attempt % 5 == 1:
                    self._log_debug(
                        "Attempt %s - backend PID %s running",
                        attempt,
                        self.backend_proc.pid,
                    )

            for url in urls:
                try:
                    if attempt <= 3:
                        self._log_debug("Trying: %s", url)
                    response = requests.get(url, timeout=2)
                    if response.status_code == 200:
                        self._log_debug(
                            "Health check success: %s %s",
                            response.status_code,
                            url,
                        )
                        return True
                except requests.exceptions.ConnectionError:
                    if attempt <= 3:
                        self._log_debug("Server not ready")
                except requests.exceptions.Timeout:
                    if attempt <= 3:
                        self._log_debug(
                            "Request timeout (server slow to respond)",
                        )
                except requests.exceptions.RequestException as e:
                    if attempt <= 3:
                        self._log_debug(
                            "Request error: %s: %s",
                            type(e).__name__,
                            e,
                        )

            time.sleep(1)

        self._log_debug("Health check TIMEOUT after %ss", timeout)
        if self.backend_proc:
            self._log_debug(
                "Backend process status: %s",
                self.backend_proc.poll(),
            )
        return False

    def _wait_for_frontend(self, timeout: int = 30) -> bool:
        """Wait for frontend to respond to requests.

        Tries both localhost and 127.0.0.1 for maximum compatibility.

        Parameters
        ----------
        timeout : int, default=30
            Maximum seconds to wait

        Returns
        -------
        bool
            True if frontend is ready, False if timeout
        """
        # Try both localhost and 127.0.0.1 for compatibility
        urls = [
            f"http://localhost:{self.frontend_port}",
            f"http://127.0.0.1:{self.frontend_port}",
        ]
        start_time = time.time()

        while time.time() - start_time < timeout:
            for url in urls:
                try:
                    response = requests.get(url, timeout=2)
                    if response.status_code == 200:
                        return True
                except requests.exceptions.RequestException:
                    pass  # Try next URL or wait
            time.sleep(1)

        return False

    def stop(self) -> None:
        """Stop both servers gracefully and clean up ports.

        Sends SIGTERM to processes and waits up to 5 seconds for clean exit.
        If processes don't exit cleanly, kills them and cleans up ports.

        Note: Suppresses Node.js EIO errors that occur when readline interfaces
        are interrupted during shutdown (cosmetic issue, shutdown still succeeds).
        """
        if self.frontend_proc:
            try:
                # Check if already dead
                if self.frontend_proc.poll() is None:
                    # Send SIGTERM for graceful shutdown
                    self.frontend_proc.terminate()

                    # Wait briefly, suppressing any stderr noise from Node.js cleanup
                    try:
                        self.frontend_proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        # Still alive after 2s - force kill
                        self.frontend_proc.kill()
                        self.frontend_proc.wait(timeout=1)  # Wait for kill to complete
                        self._kill_port(self.frontend_port)
            except Exception:
                pass  # Best effort cleanup

        if self.backend_proc:
            try:
                # Check if already dead
                if self.backend_proc.poll() is None:
                    self.backend_proc.terminate()
                    try:
                        self.backend_proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        self.backend_proc.kill()
                        self.backend_proc.wait(timeout=1)
                        self._kill_port(self.backend_port)
            except Exception:
                pass  # Best effort cleanup

        logger.info("✅ Servers stopped")

    def _kill_port(self, port: int) -> None:
        """Kill any process listening on the specified port.

        This is a fallback cleanup method for when processes don't exit cleanly.
        """
        try:
            system = platform.system()

            if system == "Windows":
                # Windows: netstat + taskkill
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
                # Linux/macOS: lsof + kill
                subprocess.run(
                    f"lsof -ti :{port} | xargs kill -9",
                    check=False,
                    shell=True,
                    capture_output=True,
                )

            self._log_debug("Cleaned up port %s", port)
        except Exception as e:
            logger.warning("Port cleanup warning: %s", e)

    def wait(self) -> None:
        """Block until processes exit or user interrupts (Ctrl+C).

        If backend exits (e.g., via shutdown endpoint), automatically stops frontend.
        """
        try:
            # Wait for backend to exit
            if self.backend_proc:
                self.backend_proc.wait()
                logger.info("🛑 Backend process exited")

                # Backend exited - shut down frontend too
                if self.frontend_proc and self.frontend_proc.poll() is None:
                    logger.info("🛑 Shutting down frontend...")
                    self.stop()
                return

            # If no backend, just wait for frontend
            if self.frontend_proc:
                self.frontend_proc.wait()
        except KeyboardInterrupt:
            # Ctrl+C pressed - stop both servers gracefully
            logger.info("🛑 Shutting down servers...")
            self.stop()

    def _print_banner(self) -> None:
        """Print startup banner."""
        logger.info("=" * 70)
        logger.info("🚀 FairSense-AgentiX Server Launcher")
        logger.info("=" * 70)

    def _print_ready_message(self) -> None:
        """Print ready message with access URLs."""
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
    """Start FairSense-AgentiX with backend and frontend servers.

    This launches both the FastAPI backend and React UI development server,
    making the platform immediately usable in your browser.

    Parameters
    ----------
    port : int, default=8000
        Port for FastAPI backend
    ui_port : int, default=5173
        Port for React UI dev server (Vite default)
    open_browser : bool, default=True
        Automatically open browser when ready
    verbose : bool, default=True
        Stream backend/frontend logs to stdout
        If False, only show status messages (cleaner output)
    reload : bool, default=False
        Enable uvicorn auto-reload on code changes
        Useful for customizing prompts, configs, or tool parameters
        Frontend has built-in hot-reload via Vite

    Examples
    --------
    Simple usage (default ports, auto-open browser):

        >>> from fairsense_agentix import server
        >>> server.start()

    Custom ports:

        >>> server.start(port=9000, ui_port=3000)

    Clean output (status only, no logs):

        >>> server.start(verbose=False)

    Development mode with auto-reload:

        >>> server.start(reload=True)
        # Edit prompts/bias_detector.txt → backend auto-reloads

    Headless mode (no browser):

        >>> server.start(open_browser=False)

    Notes
    -----
    - Backend startup takes 30-45s on first run (model preloading)
    - Frontend auto-installs npm dependencies if needed (1-2 min first time)
    - Press Ctrl+C for graceful shutdown
    - Requires Node.js/npm installed (https://nodejs.org/)
    """
    launcher = ServerLauncher(
        backend_port=port,
        frontend_port=ui_port,
        open_browser=open_browser,
        verbose=verbose,
        reload=reload,
    )

    launcher.start()
    launcher.wait()
