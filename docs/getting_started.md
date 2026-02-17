# Getting Started with FairSense-AgentiX

This guide will walk you through installing FairSense-AgentiX, configuring it, and running your first bias analysis.

---

## Prerequisites

Before you begin, ensure you have:

- **Python 3.12+** installed
- **[uv](https://docs.astral.sh/uv/)** package manager (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- **API key** for your chosen LLM provider:
  - [Anthropic](https://console.anthropic.com/) for Claude models (recommended)
  - [OpenAI](https://platform.openai.com/api-keys) for GPT models
- **(Optional)** [Node.js 18+](https://nodejs.org/) for running the React UI

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/VectorInstitute/fairsense-AgentiX.git
cd fairsense-AgentiX
```

### 2. Set Up Virtual Environment

FairSense uses `uv` for fast, reliable dependency management:

```bash
# Sync all dependencies (includes dev tools, docs, etc.)
uv sync

# Activate the virtual environment
source .venv/bin/activate  # Linux/macOS
# OR
.venv\Scripts\activate  # Windows
```

If you only need the core runtime (no dev tools):

```bash
uv sync --no-group dev --no-group docs
```

### 3. Configure Environment Variables

Create a `.env` file in the project root using either option below.

**Option A: Copy from template (recommended)**

Use the provided `.env.example` as a starting point—it includes all supported variables and comments:

```bash
cp .env.example .env
```

Then edit `.env` and set at least the **required** values (LLM provider, model, and API key). The template documents optional settings (OCR, vision model, caching, etc.).

**Option B: Create an empty file**

```bash
touch .env
```

Then add the variables yourself. At minimum, set:

```bash
# === REQUIRED ===
FAIRSENSE_LLM_PROVIDER=anthropic  # or 'openai'
FAIRSENSE_LLM_MODEL_NAME=claude-3-5-sonnet-20241022
FAIRSENSE_LLM_API_KEY=sk-ant-your-key-here  # Your Anthropic/OpenAI API key

# === OPTIONAL ===
FAIRSENSE_OCR_TOOL=auto
FAIRSENSE_CAPTION_MODEL=auto
```

!!! tip "Configuration Priority"
    Settings are loaded in this order (highest priority first):

    1. **Shell environment variables** (highest priority)
    2. **`.env` file** in project root
    3. **Default values** in `fairsense_agentix/configs/settings.py`

See the [User Guide](user_guide.md#configuration) for full options. If changes in `.env` don't apply, see the [Configuration Guide](configuration_guide.md) for troubleshooting.

### 4. Verify Installation

Test that everything is set up correctly:

```python
# Test import
python -c "from fairsense_agentix import FairSense; print('✅ Installation successful!')"
```

---

## Your First Analysis

### Text Bias Detection

Let's start with a simple text bias analysis:

```python
from fairsense_agentix import FairSense

# Initialize the engine (loads models on first run ~30-45s)
engine = FairSense()

# Analyze a job posting for bias
text = """
We're looking for a young, energetic developer to join our
fast-paced startup team. Ideal candidates are recent college
graduates who can handle the demands of a high-pressure environment.
"""

result = engine.analyze_text(text)

# Print results
print(f"Bias Detected: {result.bias_detected}")
print(f"Risk Level: {result.risk_level}")
print(f"Summary: {result.summary}")

print(f"\nFound {len(result.bias_instances)} bias instances:")
for instance in result.bias_instances:
    print(f"  • {instance.type} ({instance.severity})")
    print(f"    Text: \"{instance.text_span}\"")
    print(f"    Reason: {instance.explanation}\n")
```

**Expected Output:**
```
Bias Detected: True
Risk Level: medium
Summary: The text contains age-related bias ("young", "recent college graduates")
         that may exclude experienced candidates...

Found 2 bias instances:
  • age (high)
    Text: "young, energetic"
    Reason: Age-related descriptors that may discourage older applicants

  • age (medium)
    Text: "recent college graduates"
    Reason: Preference for recent graduates excludes experienced professionals
```

!!! info "First Run Performance"
    The first time you run FairSense, it will download and cache:

    - Embedding models (~500MB)
    - FAISS knowledge indices (~100MB)
    - Vision models (if using image analysis, ~2GB)

    Subsequent runs are **instant** (~100-200ms startup).

### Image Bias Detection

Analyze visual content for representation issues:

```python
from fairsense_agentix import FairSense

engine = FairSense()

# Analyze an image file
with open("team_photo.jpg", "rb") as f:
    image_bytes = f.read()

result = engine.analyze_image(image_bytes)

print(f"Visual Description: {result.visual_description}")
print(f"Bias Detected: {result.bias_detected}")

for instance in result.bias_instances:
    print(f"  • {instance.type}: {instance.visual_element}")
    print(f"    {instance.explanation}")
```

### Risk Assessment (CSV/Dataset)

Evaluate ML deployment scenarios for fairness risks:

```python
from fairsense_agentix import FairSense

engine = FairSense()

# Describe your deployment scenario
scenario = """
We're deploying a resume screening model that uses GPT-4 to rank candidates.
The model is trained on historical hiring data from the past 5 years.
It will be used to filter applicants for software engineering roles.
"""

result = engine.assess_risk(scenario)

print(f"Overall Risk Level: {result.risk_level}")
print(f"\nTop Risks:")
for risk in result.risks[:5]:  # Show top 5
    print(f"  • {risk.name} (Score: {risk.score:.2f})")
    print(f"    {risk.description}")
    print(f"    Mitigation: {risk.mitigation}\n")
```

---

## Using the Web Interface

The easiest way to use FairSense is through the integrated web UI:

### Launch the Server

**Option 1: Python Script**
```python
from fairsense_agentix import server

# Start both backend and frontend
server.start()
# Opens browser automatically at http://localhost:5173
```

**Option 2: Command Line**
```bash
# Using the examples script
python examples/launch_server.py

# Or directly with Python
python -c "from fairsense_agentix import server; server.start()"

# Using uv
uv run python -c "from fairsense_agentix import server; server.start()"
```

**Option 3: Custom Ports**
```python
from fairsense_agentix import server

server.start(
    port=9000,           # Backend API port
    ui_port=3000,        # Frontend UI port
    open_browser=True,   # Auto-open browser
    verbose=True         # Show server logs
)
```

### What the Server Provides

Once running, you'll have access to:

| Component | URL | Description |
|-----------|-----|-------------|
| **React UI** | http://localhost:5173 | Interactive web interface |
| **Backend API** | http://localhost:8000 | REST API endpoints |
| **API Docs** | http://localhost:8000/docs | Interactive Swagger documentation |
| **WebSocket** | ws://localhost:8000/v1/stream/{run_id} | Real-time agent telemetry |

### Using the UI

The web interface provides:

1. **Unified Input** - Text field or drag-and-drop image upload
2. **Mode Selection** - Choose between:
   - Bias (Text) - Analyze text content
   - Bias (Image) - Analyze visual content
   - Risk - Assess deployment scenarios
3. **Live Timeline** - Watch the agent's reasoning process in real-time
4. **Results Panel** - View structured outputs with:
   - Bias instances with severity levels
   - Highlighted text/annotated images
   - Risk tables with mitigation strategies
5. **Batch Jobs** - Process multiple items at once
6. **Shutdown Button** - Gracefully stop both servers

### Shutdown the Server

**From UI:** Click the red "Shutdown" button in the top-right corner

**From Command Line:** Press `Ctrl+C` in the terminal

---

## Configuration Options

### LLM Provider Selection

FairSense supports multiple LLM backends:

```bash
# Use Claude (recommended for best results)
FAIRSENSE_LLM_PROVIDER=anthropic
FAIRSENSE_LLM_MODEL_NAME=claude-3-5-sonnet-20241022
FAIRSENSE_LLM_API_KEY=sk-ant-...

# Use GPT-4
FAIRSENSE_LLM_PROVIDER=openai
FAIRSENSE_LLM_MODEL_NAME=gpt-4
FAIRSENSE_LLM_API_KEY=sk-...

# Use local model (requires Ollama)
FAIRSENSE_LLM_PROVIDER=openai
FAIRSENSE_LLM_BASE_URL=http://localhost:11434/v1
FAIRSENSE_LLM_MODEL_NAME=llama2
```

### Tool Configuration

Control which tools the agent can use:

```bash
# OCR (text extraction from images)
FAIRSENSE_OCR_TOOL=auto           # Auto-select best available
# or: tesseract, paddleocr, fake (testing)

# Vision-Language Model (image understanding)
FAIRSENSE_CAPTION_MODEL=auto      # Auto-select best available
# or: blip2, blip, fake (testing)

# Embedding Model (semantic search)
FAIRSENSE_EMBEDDING_PROVIDER=auto
# or: sentence-transformers, openai
```

### Refinement & Evaluation

Enable/disable the iterative refinement loop:

```bash
# Enable agent self-critique and refinement
FAIRSENSE_ENABLE_REFINEMENT=true
FAIRSENSE_EVALUATOR_ENABLED=true

# Set quality thresholds (0-100)
FAIRSENSE_BIAS_EVALUATOR_MIN_SCORE=75   # Minimum passing score
FAIRSENSE_MAX_REFINEMENT_ITERATIONS=2    # Limit refinement cycles
```

!!! warning "Performance vs. Quality Trade-off"
    - **Refinement ON** (default): Slower but higher quality outputs (~2-3 min per analysis)
    - **Refinement OFF**: Faster but may miss edge cases (~30-60s per analysis)

    For production use, we recommend keeping refinement enabled.

---

## Troubleshooting

### API Key Issues

**Symptom:** `AuthenticationError` or `401 Unauthorized`

**Solution:**
```bash
# Verify your key is set correctly
echo $FAIRSENSE_LLM_API_KEY  # Should show your key

# If empty, set it:
export FAIRSENSE_LLM_API_KEY=your-key-here

# Or add to .env file
echo "FAIRSENSE_LLM_API_KEY=your-key-here" >> .env
```

### Model Download Timeouts

**Symptom:** First run hangs for 5+ minutes

**Cause:** Downloading large embedding/vision models

**Solution:**
1. Be patient - models only download once (~2GB total)
2. Check your internet connection
3. Models are cached in `~/.cache/huggingface/`

### Port Already in Use

**Symptom:** `Address already in use` error

**Solution:**
```bash
# Find process using port 8000
lsof -ti :8000 | xargs kill -9  # Linux/macOS
netstat -ano | findstr :8000    # Windows (then taskkill)

# Or use custom ports
server.start(port=9000, ui_port=3000)
```

### Memory Issues

**Symptom:** `OutOfMemoryError` or system freezes

**Solution:**
```bash
# Use lighter models
FAIRSENSE_CAPTION_MODEL=fake  # Skip vision model loading
FAIRSENSE_OCR_TOOL=tesseract   # Lighter than PaddleOCR

# Or disable refinement to reduce LLM calls
FAIRSENSE_ENABLE_REFINEMENT=false
```

### Import Errors

**Symptom:** `ModuleNotFoundError: No module named 'fairsense_agentix'`

**Solution:**
```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Reinstall dependencies
uv sync

# Verify installation
python -c "import fairsense_agentix; print('✅ Installed')"
```

---

## Next Steps

Now that you have FairSense running, explore:

- **[User Guide](user_guide.md)** - Detailed examples for each workflow (text, image, risk)
- **[API Reference](api.md)** - Full Python API and REST endpoint documentation
- **[Server Guide](server.md)** - Deployment and production setup

---

## Getting Help

- **GitHub Issues**: [Report bugs](https://github.com/VectorInstitute/fairsense-AgentiX/issues)
- **Discussions**: [Ask questions](https://github.com/VectorInstitute/fairsense-AgentiX/discussions)
- **Documentation**: You're reading it! 📚
