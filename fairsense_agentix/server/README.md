# FairSense-AgentiX Server Launcher

Programmatic server launcher for starting both the FastAPI backend and React frontend with a single function call.

## Quick Start

```python
from fairsense_agentix import server

server.start()
```

This will:
- ▶️ Start FastAPI backend on http://localhost:8000
- ▶️ Start React UI on http://localhost:5173
- 🌐 Auto-open your browser
- ⏳ Wait for both services to be ready (health checks)
- 🛑 Gracefully shutdown on Ctrl+C

## Usage Examples

### Basic Usage

```python
from fairsense_agentix import server

# Default configuration
server.start()
```

**Output:**
```
======================================================================
🚀 FairSense-AgentiX Server Launcher
======================================================================

▶️  Starting backend on port 8000...
⏳ Waiting for backend to be ready...
✅ Backend ready

▶️  Starting frontend on port 5173...
⏳ Waiting for frontend to be ready...
✅ Frontend ready

🌐 Opening browser at http://localhost:5173

======================================================================
✅ FairSense-AgentiX is running!
======================================================================
Backend:  http://localhost:8000
Frontend: http://localhost:5173
API Docs: http://localhost:8000/docs

Press Ctrl+C to stop
======================================================================
```

### Custom Ports

```python
server.start(port=9000, ui_port=3000)
```

### Clean Output (Status Only)

```python
server.start(verbose=False)
```

Suppresses backend/frontend logs, shows only status messages.

### Development Mode (Auto-Reload)

```python
server.start(reload=True)
```

Backend auto-reloads when you:
- Edit prompts in `prompts/`
- Modify configs in `.env`
- Change tool parameters

**Use case:** Researchers iterating on bias detection prompts

### Headless Mode (No Browser)

```python
server.start(open_browser=False)
```

## Configuration Options

```python
def start(
    port: int = 8000,          # FastAPI backend port
    ui_port: int = 5173,       # React UI port (Vite default)
    open_browser: bool = True, # Auto-open browser
    verbose: bool = True,      # Stream logs to stdout
    reload: bool = False       # Enable auto-reload
) -> None:
```

## Requirements

- **Python 3.12+** with fairsense-agentix package installed
- **Node.js 18+** with npm (for React UI)
- **Dependencies:** `uv sync` or `pip install -e .`

## Troubleshooting

### Error: `npm not found`

**Solution:** Install Node.js

```bash
# macOS
brew install node

# Windows
https://nodejs.org/

# Linux
sudo apt install nodejs npm
```

### Error: `Backend failed to start on port 8000`

**Cause:** Port already in use

**Solution:** Check what's using the port

```bash
# macOS/Linux
lsof -i :8000

# Windows
netstat -ano | findstr :8000
```

Then either:
- Kill the process using that port
- Use a different port: `server.start(port=9000)`

### Error: `Frontend failed to start on port 5173`

**Solution 1:** Install npm dependencies

```bash
cd ui
npm install
```

**Solution 2:** Use different port

```python
server.start(ui_port=3000)
```

### Backend Startup is Slow (30-45s)

This is normal on first run! Backend preloads:
- LLM models (OpenAI/Anthropic clients)
- Embedding models (~90MB download, cached)
- FAISS indexes (~5MB)
- OCR/Caption models (~1GB, cached)

Subsequent starts are much faster (~5-10s).

## Architecture

### Subprocess-Based Design

The launcher uses Python's `subprocess.Popen` to start:

1. **Backend:** `python run_server.py` (uvicorn + FastAPI)
2. **Frontend:** `npm run dev` in `ui/` (Vite + React)

Both run as separate OS processes with:
- ✅ Process isolation (separate stdout/stderr)
- ✅ Graceful shutdown (SIGTERM/SIGINT)
- ✅ Health checking (HTTP polling)
- ✅ Cross-platform support (Windows/Linux/macOS)

### Health Checking

**Backend:**
- Polls `GET /v1/health` every 1s
- Timeout: 60s (accounts for model loading)
- Ready when: HTTP 200 OK

**Frontend:**
- Polls `GET http://localhost:5173` every 1s
- Timeout: 30s
- Ready when: HTTP 200 OK

### Shutdown Sequence

1. User presses Ctrl+C (SIGINT)
2. Signal handler catches SIGINT
3. Terminate frontend process (`.terminate()`)
4. Wait up to 5s for frontend exit
5. Terminate backend process (`.terminate()`)
6. Wait up to 5s for backend exit
7. Exit with status 0

## Testing

Run unit tests:

```bash
pytest tests/test_server_launcher.py -v
```

Manual integration test:

```bash
python examples/launch_server.py
```

Then:
1. Verify browser opens automatically
2. Check UI loads and is responsive
3. Test a text bias analysis
4. Press Ctrl+C
5. Verify both servers stop cleanly

## Module Structure

```
fairsense_agentix/server/
├── __init__.py           # Public API exports
├── launcher.py           # ServerLauncher class + start() function
└── README.md             # This file
```

## See Also

- [Beta Implementation Plan](../../../planning_files/beta_implementation_plan.md)
- [API Documentation](http://localhost:8000/docs) (when running)
- [Project README](../../../README.md)
