# FairSense - AgentiX

----------------------------------------------------------------------------------------

[![code checks](https://github.com/VectorInstitute/fairsense-AgentiX/actions/workflows/code_checks.yml/badge.svg)](https://github.com/VectorInstitute/fairsense-AgentiX/actions/workflows/code_checks.yml)
[![integration tests](https://github.com/VectorInstitute/fairsense-AgentiX/actions/workflows/integration_tests.yml/badge.svg)](https://github.com/VectorInstitute/fairsense-AgentiX/actions/workflows/integration_tests.yml)
[![docs](https://github.com/VectorInstitute/fairsense-AgentiX/actions/workflows/docs.yml/badge.svg)](https://github.com/VectorInstitute/fairsense-AgentiX/actions/workflows/docs.yml)
[![codecov](https://codecov.io/github/VectorInstitute/fairsense-AgentiX/graph/badge.svg?token=83MYFZ3UPA)](https://codecov.io/github/VectorInstitute/fairsense-AgentiX)

<!-- TODO: Uncomment this with the right credentials once codecov is set up for this repo.
[![codecov](https://codecov.io/github/VectorInstitute/fairsense-AgentiX/graph/badge.svg?token=83MYFZ3UPA)](https://codecov.io/github/VectorInstitute/fairsense-AgentiX)
-->
<!-- TODO: Uncomment this when the repository is made public
![GitHub License](https://img.shields.io/github/license/VectorInstitute/fairsense-AgentiX)
-->

<!--
TODO: Add picture / logo
-->

<!--
TODO: Add introduction about Fairsense-AgentiX here
-->

## 🧑🏿‍💻 Developing

### Installing dependencies

The development environment can be set up using
[uv](https://github.com/astral-sh/uv?tab=readme-ov-file#installation). Hence, make sure it is
installed and then run:

```bash
uv sync
source .venv/bin/activate
```

In order to install dependencies for testing (codestyle, unit tests, integration tests),
run:

```bash
uv sync --dev
source .venv/bin/activate
```

In order to exclude installation of packages from a specific group (e.g. docs),
run:

```bash
uv sync --no-group docs
```
## Getting Started

### Run the FastAPI service (Phase 9)

```bash
uv run uvicorn fairsense_agentix.service_api.server:app --reload
```

Endpoints (all under `/v1/...`):

| Route | Description |
| --- | --- |
| `POST /analyze` | JSON payload with `content`, optional `input_type`, `options` |
| `POST /analyze/upload` | `multipart/form-data` for images |
| `POST /batch` & `GET /batch/{id}` | Submit + inspect batch jobs |
| `GET /health` | Health probe |
| `WS /stream/{run_id}` | Stream telemetry/agent events for a run |

The API auto-detects text/image/CSV inputs, but you can override by setting `input_type` to `bias_text`, `bias_image`, or `risk`.

### Run the Claude-inspired UI (Phase 11)

```bash
cd ui
npm install
npm run dev
```

Set `VITE_API_BASE` (defaults to `http://localhost:8000`) to point at the API. The UI provides:

- Unified input surface (text field + drag/drop image upload)
- Live agent timeline sourced from telemetry events
- Downloadable HTML highlights and risk tables
- Launchpad for batch jobs

### Key configuration knobs

Configure via environment variables (see `.env` for the full list). Most relevant:

| Variable | Description |
| --- | --- |
| `FAIRSENSE_LLM_PROVIDER` | `openai`, `anthropic`, or `fake` |
| `FAIRSENSE_LLM_MODEL_NAME` | e.g. `gpt-4`, `claude-3-5-sonnet` |
| `FAIRSENSE_LLM_API_KEY` | Provider API key |
| `FAIRSENSE_OCR_TOOL` | `auto`, `tesseract`, `paddleocr`, `fake` |
| `FAIRSENSE_CAPTION_MODEL` | `auto`, `blip2`, `blip`, `fake` |
| `FAIRSENSE_ENABLE_REFINEMENT` | enables evaluator-driven retries (default `true`) |
| `FAIRSENSE_EVALUATOR_ENABLED` | toggles Phase 7 evaluators |
| `FAIRSENSE_BIAS_EVALUATOR_MIN_SCORE` | passing score (0–100, default 75) |

All settings can be overridden at runtime:

```bash
FAIRSENSE_LLM_PROVIDER=anthropic \
FAIRSENSE_LLM_MODEL_NAME=claude-3-5-sonnet-20241022 \
uv run uvicorn fairsense_agentix.service_api.server:app
```

## Features / Components

## Examples

## Contributing
If you are interested in contributing to the library, please see
[CONTRIBUTING.MD](CONTRIBUTING.MD). This file contains many details around contributing
to the code base, including development practices, code checks, tests, and more.

<!--
TODO:

## Acknowledgements

## Citation

-->
