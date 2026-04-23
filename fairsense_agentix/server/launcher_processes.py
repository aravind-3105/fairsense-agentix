"""Subprocess startup for backend (uvicorn) and frontend (Vite)."""

import logging
import os
import subprocess
import sys
from pathlib import Path


logger = logging.getLogger("fairsense_agentix.server.launcher")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _log_debug(verbose: bool, msg: str, *args: object) -> None:
    if verbose:
        logger.debug(msg, *args)


def start_backend_process(
    backend_port: int,
    reload: bool,
    verbose: bool,
) -> subprocess.Popen:
    """Start FastAPI backend via ``run_server.py``."""
    env = os.environ.copy()
    env["FAIRSENSE_API_PORT"] = str(backend_port)
    env["FAIRSENSE_API_RELOAD"] = "true" if reload else "false"

    if "FAIRSENSE_DISABLE_EAGER_LOADING" in env:
        del env["FAIRSENSE_DISABLE_EAGER_LOADING"]
        _log_debug(verbose, "Re-enabled eager loading for backend subprocess")

    script = _PROJECT_ROOT / "run_server.py"

    _log_debug(verbose, "Backend script path: %s", script)
    _log_debug(verbose, "Script exists: %s", script.exists())
    _log_debug(verbose, "Project root: %s", _PROJECT_ROOT)
    _log_debug(verbose, "Python executable: %s", sys.executable)
    _log_debug(verbose, "Backend port: %s", backend_port)

    if not script.exists():
        raise FileNotFoundError(
            f"Backend launcher script not found: {script}\n"
            f"Ensure you're running from the project root directory.",
        )

    _log_debug(verbose, "Starting backend subprocess...")
    cmd = [sys.executable, str(script)]
    cwd = str(_PROJECT_ROOT)

    if verbose:
        proc = subprocess.Popen(cmd, env=env, cwd=cwd)
    else:
        proc = subprocess.Popen(
            cmd,
            env=env,
            cwd=cwd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    _log_debug(verbose, "Backend process started with PID: %s", proc.pid)
    _log_debug(verbose, "Process poll status: %s", proc.poll())
    return proc


def start_frontend_process(
    backend_port: int,
    frontend_port: int,
    verbose: bool,
) -> subprocess.Popen:
    """Start Vite dev server under ``ui/``."""
    env = os.environ.copy()
    env["VITE_API_BASE"] = f"http://localhost:{backend_port}"

    ui_dir = _PROJECT_ROOT / "ui"

    if not ui_dir.exists():
        raise FileNotFoundError(
            f"UI directory not found: {ui_dir}\n"
            f"Ensure you cloned the repository completely.",
        )

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
            capture_output=not verbose,
        )
        if install_proc.returncode != 0:
            raise RuntimeError(
                "npm install failed. Check your internet connection and try again.",
            )

    npm_cmd = [
        "npm",
        "run",
        "dev",
        "--",
        "--port",
        str(frontend_port),
        "--strictPort",
    ]

    if verbose:
        proc = subprocess.Popen(
            npm_cmd,
            cwd=str(ui_dir),
            env=env,
        )
    else:
        proc = subprocess.Popen(
            npm_cmd,
            cwd=str(ui_dir),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    _log_debug(verbose, "Frontend process started with PID: %s", proc.pid)
    return proc
