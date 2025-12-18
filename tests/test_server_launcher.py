"""Tests for server launcher module.

These tests verify the ServerLauncher class initialization and configuration
without actually starting real server processes (which would require mocking
subprocess and network calls).
"""

from fairsense_agentix.server.launcher import ServerLauncher


def test_launcher_initialization_defaults():
    """Test ServerLauncher initializes with default parameters."""
    launcher = ServerLauncher()

    assert launcher.backend_port == 8000
    assert launcher.frontend_port == 5173
    assert launcher.open_browser is True
    assert launcher.verbose is True
    assert launcher.reload is False
    assert launcher.backend_proc is None
    assert launcher.frontend_proc is None


def test_launcher_initialization_custom():
    """Test ServerLauncher accepts custom parameters."""
    launcher = ServerLauncher(
        backend_port=9000,
        frontend_port=3000,
        open_browser=False,
        verbose=False,
        reload=True,
    )

    assert launcher.backend_port == 9000
    assert launcher.frontend_port == 3000
    assert launcher.open_browser is False
    assert launcher.verbose is False
    assert launcher.reload is True


def test_launcher_stop_with_no_processes():
    """Test stop() handles case where no processes were started."""
    launcher = ServerLauncher()

    # Should not raise error even if no processes exist
    launcher.stop()

    assert launcher.backend_proc is None
    assert launcher.frontend_proc is None


def test_server_module_exports():
    """Test that server module exports start function."""
    from fairsense_agentix import server

    assert hasattr(server, "start")
    assert callable(server.start)


def test_start_function_signature():
    """Test start() function has correct signature."""
    import inspect

    from fairsense_agentix.server import start

    sig = inspect.signature(start)
    params = sig.parameters

    # Check expected parameters exist
    assert "port" in params
    assert "ui_port" in params
    assert "open_browser" in params
    assert "verbose" in params
    assert "reload" in params

    # Check defaults
    assert params["port"].default == 8000
    assert params["ui_port"].default == 5173
    assert params["open_browser"].default is True
    assert params["verbose"].default is True
    assert params["reload"].default is False


# Integration tests (require mocking subprocess and requests)
# These would be implemented in a separate integration test suite
# that uses pytest-mock to mock subprocess.Popen and requests.get

# Example integration tests (not implemented here):
#
# def test_backend_startup_success(mocker):
#     # Mock subprocess.Popen for backend
#     # Mock requests.get to return 200 for health check
#     # Assert backend process started correctly
#     pass
#
# def test_frontend_startup_success(mocker):
#     # Mock subprocess.Popen for frontend
#     # Mock requests.get to return 200 for frontend
#     # Assert frontend process started correctly
#     pass
#
# def test_backend_health_check_timeout(mocker):
#     # Mock requests.get to always timeout
#     # Assert launcher fails gracefully
#     pass
#
# def test_npm_not_found(mocker):
#     # Mock subprocess.Popen to raise FileNotFoundError
#     # Assert appropriate error message
#     pass
#
# def test_graceful_shutdown(mocker):
#     # Start both processes (mocked)
#     # Call stop()
#     # Assert terminate() called on both processes
#     pass
