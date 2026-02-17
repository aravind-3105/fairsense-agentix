# FairSense-AgentiX Configuration Guide

## Problem: Configuration Not Loading from .env

### Symptoms
- Changed values in `.env` file not reflected when running programs
- `llm_provider` shows "fake" even though `.env` has "openai"
- API key not loaded even though set in `.env`

### Root Cause
Shell environment variables override `.env` file values. This is by design in Pydantic Settings:

**Priority Order (highest to lowest):**
1. Environment variables in your shell (`export FAIRSENSE_LLM_PROVIDER=openai`)
2. `.env` file values (`FAIRSENSE_LLM_PROVIDER=openai`)
3. Code defaults (`llm_provider="fake"`)

### How It Happened
At some point, FAIRSENSE_ variables were exported to your shell environment and remain there.

### Solution

**Option 1: New terminal (Recommended)**  
Open a fresh terminal. Environment variables don't persist across terminals, so a new session will read only from `.env` (and any new exports you add there).

**Option 2: Manual cleanup**
```bash
# One-time: Clear all FAIRSENSE_ variables in this shell
for var in $(env | grep '^FAIRSENSE_' | cut -d= -f1); do unset $var; done

# Then run your program (e.g. Python API or server)
uv run python -c "from fairsense_agentix import FairSense; print('OK')"
```

**Option 3: Override in the same command**  
Run with explicit env for that invocation only:
```bash
FAIRSENSE_LLM_PROVIDER=openai FAIRSENSE_LLM_API_KEY=your-key uv run python -c "from fairsense_agentix import FairSense; FairSense()"
```

### Verification
In a **new terminal** (so shell env doesn't override), run:
```bash
uv run python -c "
from fairsense_agentix.configs.settings import Settings
s = Settings()
print('SUCCESS: .env is being used' if (s.llm_provider != 'fake' or s.llm_api_key) else 'Check .env or set FAIRSENSE_LLM_PROVIDER and FAIRSENSE_LLM_API_KEY')
"
```
If you see "SUCCESS", `.env` is being read. If values still look wrong, ensure you didn't export `FAIRSENSE_*` in this shell.

## Configuration Best Practices

### For Development
Use `.env` file for local configuration:
```bash
# .env
FAIRSENSE_LLM_PROVIDER=openai
FAIRSENSE_LLM_API_KEY=sk-...
FAIRSENSE_OCR_TOOL=auto
```

Don't export to shell unless you need to override temporarily.

### For Production/Deployment
Use environment variables (they override `.env` by design):
```bash
export FAIRSENSE_LLM_PROVIDER=openai
export FAIRSENSE_LLM_API_KEY=sk-proj-...
docker run -e FAIRSENSE_LLM_PROVIDER=openai myapp
```

### For Testing
Override specific settings without editing `.env`:
```bash
FAIRSENSE_LLM_PROVIDER=fake uv run pytest
```

## Files
- `.env` - Your local configuration (gitignored; create from `.env.example`)
- `fairsense_agentix/configs/settings.py` - Settings class and defaults

## Technical Details

Pydantic Settings loads configuration in this order:
1. `Settings()` constructor arguments
2. Shell environment variables
3. `.env` file (if `env_file` configured)
4. Field defaults in code

The `.env` file IS being read - you can verify by checking import errors when you remove it. Environment variables just take precedence.
