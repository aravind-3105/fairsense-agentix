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

**Option 1: Use the wrapper script (Recommended)**
```bash
./run_demo.sh
```
This automatically clears environment variables and reads from `.env`.

**Option 2: Manual cleanup**
```bash
# One-time: Clear all FAIRSENSE_ variables
for var in $(env | grep '^FAIRSENSE_' | cut -d= -f1); do unset $var; done

# Then run your program
uv run python planning_files/test_demo.py
```

**Option 3: New terminal**
Open a fresh terminal (environment variables don't persist across terminals).

### Verification
Test if `.env` is being read:
```bash
uv run python test_config.py
```

This will show:
- ✓ SUCCESS: Configuration loaded from .env file! (good)
- ✗ PROBLEM: Environment variables are overriding .env file (need to unset)

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
- `.env` - Your local configuration (gitignored)
- `test_config.py` - Test configuration loading
- `run_demo.sh` - Wrapper that handles env cleanup
- `fairsense_agentix/configs/settings.py` - Settings class

## Technical Details

Pydantic Settings loads configuration in this order:
1. `Settings()` constructor arguments
2. Shell environment variables
3. `.env` file (if `env_file` configured)
4. Field defaults in code

The `.env` file IS being read - you can verify by checking import errors when you remove it. Environment variables just take precedence.
