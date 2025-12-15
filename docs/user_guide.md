# User Guide

This guide provides detailed examples and best practices for using FairSense-AgentiX in your applications.

---

## Overview

FairSense-AgentiX provides three main analysis workflows:

1. **Text Bias Detection** - Analyze written content for fairness issues
2. **Image Bias Detection** - Analyze visual content for representation problems
3. **Risk Assessment** - Evaluate ML deployment scenarios for compliance risks

Each workflow uses an **agentic reasoning system** that:
- Plans its analysis strategy
- Selects appropriate tools dynamically
- Iteratively refines outputs based on self-critique
- Provides full transparency via event telemetry

---

## Text Bias Analysis

### Basic Usage

```python
from fairsense_agentix import FairSense

engine = FairSense()

text = """
We need a rockstar developer who's a cultural fit for our young,
dynamic team. Must be willing to work long hours and weekends.
"""

result = engine.analyze_text(text)

# Access results
print(f"Bias Detected: {result.bias_detected}")
print(f"Risk Level: {result.risk_level}")
print(f"Summary: {result.summary}")

# Iterate through bias instances
for instance in result.bias_instances:
    print(f"\nType: {instance.type}")
    print(f"Severity: {instance.severity}")
    print(f"Text: '{instance.text_span}'")
    print(f"Explanation: {instance.explanation}")
    print(f"Mitigation: {instance.mitigation_suggestion}")
```

### Understanding the Result Object

The `BiasResult` object contains:

| Field | Type | Description |
|-------|------|-------------|
| `bias_detected` | `bool` | Whether any bias was found |
| `risk_level` | `str` | Overall severity: `low`, `medium`, `high` |
| `summary` | `str` | High-level explanation of findings |
| `bias_instances` | `list[BiasInstance]` | Detailed bias findings |
| `highlighted_html` | `str` | HTML with color-coded highlights |
| `metadata` | `ResultMetadata` | Execution details (time, model, workflow ID) |

**BiasInstance Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `type` | `str` | Bias category: `gender`, `age`, `racial`, `disability`, `socioeconomic` |
| `severity` | `str` | Impact level: `low`, `medium`, `high` |
| `text_span` | `str` | The problematic text excerpt |
| `explanation` | `str` | Why this is biased |
| `mitigation_suggestion` | `str` | How to fix it |
| `context` | `str` | Surrounding text for context |

### Advanced: Highlighted HTML Output

FairSense generates color-coded HTML for easy visualization:

```python
result = engine.analyze_text(text)

# Save highlighted version
with open("analysis_output.html", "w") as f:
    f.write(result.highlighted_html)

# Or display in Jupyter
from IPython.display import HTML
HTML(result.highlighted_html)
```

The HTML uses color coding:

- 🟥 **Red** - Gender bias
- 🟧 **Orange** - Age bias
- 🟨 **Yellow** - Racial bias
- 🟦 **Blue** - Disability bias
- 🟪 **Purple** - Socioeconomic bias

### Common Use Cases

#### Job Postings

```python
job_posting = """
Job Title: Senior Software Engineer

Requirements:
- Recent CS graduate preferred
- Native English speaker
- Must be able to work in a fast-paced environment
- Looking for someone who's a culture fit
"""

result = engine.analyze_text(job_posting)

# Check for hiring-related biases
for instance in result.bias_instances:
    if instance.type in ["age", "racial"]:
        print(f"⚠️  Legal Risk: {instance.text_span}")
        print(f"   Mitigation: {instance.mitigation_suggestion}")
```

#### Marketing Copy

```python
ad_copy = """
Our product is designed for busy moms who need quick solutions
to help them manage their households while looking their best.
"""

result = engine.analyze_text(ad_copy)

# Identify gender stereotyping
gender_biases = [i for i in result.bias_instances if i.type == "gender"]
print(f"Found {len(gender_biases)} gender-related issues")
```

#### Content Moderation

```python
user_comment = """
This feature is so easy to use, even my grandma could figure it out!
"""

result = engine.analyze_text(user_comment)

if result.bias_detected:
    print(f"⚠️  Flagged for review: {result.summary}")
```

---

## Image Bias Analysis

### Basic Usage

```python
from fairsense_agentix import FairSense

engine = FairSense()

# Load image file
with open("team_photo.jpg", "rb") as f:
    image_bytes = f.read()

result = engine.analyze_image(image_bytes)

# Access results
print(f"Visual Description: {result.visual_description}")
print(f"Bias Detected: {result.bias_detected}")
print(f"Risk Level: {result.risk_level}")

# Bias instances for images
for instance in result.bias_instances:
    print(f"\nType: {instance.type}")
    print(f"Visual Element: {instance.visual_element}")
    print(f"Explanation: {instance.explanation}")
```

### Image Result Object

The image analysis result includes:

| Field | Type | Description |
|-------|------|-------------|
| `visual_description` | `str` | What the VLM sees in the image |
| `bias_detected` | `bool` | Whether bias was found |
| `bias_instances` | `list[BiasInstance]` | Visual bias findings |
| `image_base64` | `str` | Base64-encoded annotated image |
| `reasoning_trace` | `str` | Agent's step-by-step reasoning |

**BiasInstance for Images:**

| Field | Type | Description |
|-------|------|-------------|
| `type` | `str` | Bias category |
| `visual_element` | `str` | Description of problematic element |
| `explanation` | `str` | Why this represents bias |
| `severity` | `str` | Impact level |

### Working with Different Image Sources

#### From File

```python
with open("advertisement.png", "rb") as f:
    result = engine.analyze_image(f.read())
```

#### From URL

```python
import requests

url = "https://example.com/marketing-image.jpg"
image_bytes = requests.get(url).content
result = engine.analyze_image(image_bytes)
```

#### From PIL Image

```python
from PIL import Image
import io

# Load and process PIL image
pil_image = Image.open("photo.jpg")

# Convert to bytes
buffer = io.BytesIO()
pil_image.save(buffer, format="JPEG")
image_bytes = buffer.getvalue()

result = engine.analyze_image(image_bytes)
```

### Common Use Cases

#### Stock Photo Auditing

```python
import os

# Analyze a directory of stock photos
for filename in os.listdir("stock_photos/"):
    if filename.endswith((".jpg", ".png")):
        with open(f"stock_photos/{filename}", "rb") as f:
            result = engine.analyze_image(f.read())

        if result.bias_detected:
            print(f"\n⚠️  {filename}")
            print(f"   Risk: {result.risk_level}")
            print(f"   Issue: {result.summary}")
```

#### Marketing Material Review

```python
# Check ad campaign images
campaign_images = ["ad_1.jpg", "ad_2.jpg", "ad_3.jpg"]

issues = []
for img_path in campaign_images:
    with open(img_path, "rb") as f:
        result = engine.analyze_image(f.read())

    if result.risk_level in ["medium", "high"]:
        issues.append({
            "file": img_path,
            "issues": result.bias_instances,
            "description": result.visual_description
        })

# Generate report
if issues:
    print(f"Found {len(issues)} images needing review")
```

---

## Risk Assessment

### Basic Usage

```python
from fairsense_agentix import FairSense

engine = FairSense()

scenario = """
We're deploying a loan approval system that uses historical data
from the past 10 years. The model uses applicant income, credit score,
employment history, and zip code to predict default risk.
"""

result = engine.assess_risk(scenario)

# Access results
print(f"Overall Risk: {result.risk_level}")
print(f"Total Risks Identified: {len(result.risks)}")

# Top risks
for risk in result.risks[:5]:
    print(f"\n{risk.name} (Score: {risk.score:.2f}/100)")
    print(f"  Category: {risk.category}")
    print(f"  Description: {risk.description}")
    print(f"  Mitigation: {risk.mitigation}")
```

### Risk Result Object

| Field | Type | Description |
|-------|------|-------------|
| `risk_level` | `str` | Overall severity: `low`, `medium`, `high`, `critical` |
| `risks` | `list[RiskItem]` | Identified risks, sorted by score |
| `summary` | `str` | High-level risk overview |
| `metadata` | `ResultMetadata` | Execution details |

**RiskItem Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Risk identifier |
| `category` | `str` | Risk type: `fairness`, `security`, `compliance`, etc. |
| `score` | `float` | Severity score (0-100) |
| `description` | `str` | What the risk is |
| `mitigation` | `str` | How to address it |
| `related_framework` | `str` | AI-RMF, NIST, etc. |

### CSV Dataset Analysis

For structured data, you can provide CSV content directly:

```python
csv_data = """
age,gender,income,approved
25,M,50000,1
45,F,60000,0
30,M,55000,1
"""

# Describe the context
context = """
This is training data for a loan approval model. The 'approved' column
indicates whether the loan was historically approved (1) or denied (0).
"""

result = engine.assess_risk(context + "\n\n" + csv_data)

# Check for fairness risks
fairness_risks = [r for r in result.risks if r.category == "fairness"]
print(f"Found {len(fairness_risks)} fairness concerns")
```

### Common Use Cases

#### Pre-Deployment Audit

```python
deployment_plan = """
Model: GPT-4 fine-tuned on customer service transcripts
Use Case: Automated response generation for support tickets
Training Data: 500k customer interactions from 2020-2024
Deployment: Production API serving 10k requests/day
Monitoring: None currently planned
"""

result = engine.assess_risk(deployment_plan)

# Check critical risks
critical = [r for r in result.risks if r.score > 80]
if critical:
    print("🚨 CRITICAL RISKS - DO NOT DEPLOY")
    for risk in critical:
        print(f"  • {risk.name}: {risk.description}")
```

#### Compliance Check

```python
# Check against regulatory frameworks
result = engine.assess_risk(scenario)

# Filter by compliance category
compliance_risks = [r for r in result.risks if r.category == "compliance"]

for risk in compliance_risks:
    print(f"\n⚖️  {risk.name}")
    print(f"   Framework: {risk.related_framework}")
    print(f"   Mitigation: {risk.mitigation}")
```

#### Vendor Assessment

```python
vendor_proposal = """
Vendor: ThirdParty ML Inc.
Product: Resume Screening AI
Claims: 95% accuracy, reduces hiring time by 60%
Training Data: Proprietary dataset (details not disclosed)
Explainability: Black-box model
"""

result = engine.assess_risk(vendor_proposal)

# Generate vendor scorecard
print(f"Vendor Risk Score: {sum(r.score for r in result.risks) / len(result.risks):.1f}/100")
print(f"High-Priority Concerns: {len([r for r in result.risks if r.score > 70])}")
```

---

## Configuration

### Environment Variables

All settings can be configured via environment variables or the `.env` file:

```bash
# === Core LLM Settings ===
FAIRSENSE_LLM_PROVIDER=anthropic
FAIRSENSE_LLM_MODEL_NAME=claude-3-5-sonnet-20241022
FAIRSENSE_LLM_API_KEY=your-key-here
FAIRSENSE_LLM_TEMPERATURE=0.7
FAIRSENSE_LLM_MAX_TOKENS=4096

# === Tool Selection ===
FAIRSENSE_OCR_TOOL=auto              # tesseract, paddleocr, fake
FAIRSENSE_CAPTION_MODEL=auto         # blip, blip2, fake
FAIRSENSE_EMBEDDING_PROVIDER=auto    # sentence-transformers, openai

# === Agent Behavior ===
FAIRSENSE_ENABLE_REFINEMENT=true     # Enable self-critique loop
FAIRSENSE_MAX_REFINEMENT_ITERATIONS=2
FAIRSENSE_EVALUATOR_ENABLED=true
FAIRSENSE_BIAS_EVALUATOR_MIN_SCORE=75

# === Performance ===
FAIRSENSE_WORKFLOW_TIMEOUT_SECONDS=300
FAIRSENSE_CACHE_ENABLED=true
FAIRSENSE_CACHE_TTL_SECONDS=3600

# === Server Settings ===
FAIRSENSE_API_HOST=0.0.0.0
FAIRSENSE_API_PORT=8000
FAIRSENSE_API_RELOAD=false
```

### Programmatic Configuration

Override settings in code:

```python
from fairsense_agentix import FairSense
from fairsense_agentix.configs.settings import Settings

# Create custom settings
settings = Settings(
    llm_provider="anthropic",
    llm_model_name="claude-3-5-sonnet-20241022",
    enable_refinement=False,  # Faster but lower quality
    max_refinement_iterations=1
)

# Initialize engine with custom settings
engine = FairSense(settings=settings)
```

### Per-Analysis Options

Pass options to individual analysis calls:

```python
# Disable refinement for this specific analysis
result = engine.analyze_text(
    text,
    enable_refinement=False,  # Skip self-critique
    max_refinement_iterations=0
)

# Adjust LLM temperature
result = engine.analyze_text(
    text,
    llm_temperature=0.3  # More deterministic
)
```

---

## Batch Processing

Process multiple items efficiently using the batch API:

### Using the Python API

```python
from fairsense_agentix import FairSense

engine = FairSense()

# Prepare batch items
texts = [
    "Job posting 1...",
    "Job posting 2...",
    "Job posting 3...",
]

# Process in batch (runs sequentially with progress tracking)
results = []
for i, text in enumerate(texts, 1):
    print(f"Processing {i}/{len(texts)}...")
    result = engine.analyze_text(text)
    results.append(result)

# Aggregate results
biased_count = sum(1 for r in results if r.bias_detected)
print(f"\n{biased_count}/{len(results)} items flagged")
```

### Using the REST API

```python
import requests
import time

# Submit batch job
batch_request = {
    "items": [
        {"content": "Text 1", "input_type": "bias_text"},
        {"content": "Text 2", "input_type": "bias_text"},
        {"content": "Text 3", "input_type": "bias_text"},
    ]
}

response = requests.post(
    "http://localhost:8000/v1/batch",
    json=batch_request
)
job_id = response.json()["job_id"]

# Poll for completion
while True:
    status_response = requests.get(
        f"http://localhost:8000/v1/batch/{job_id}"
    )
    status = status_response.json()

    print(f"Progress: {status['completed']}/{status['total']}")

    if status["status"] in ["completed", "failed"]:
        break

    time.sleep(5)

# Access results
results = status["results"]
```

---

## REST API Usage

### Analyze Text

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/analyze",
    json={
        "content": "Your text here...",
        "input_type": "bias_text",  # Optional: auto-detected if omitted
        "options": {}  # Optional configuration
    }
)

result = response.json()
print(result["bias_detected"])
```

### Analyze Image (File Upload)

```python
import requests

with open("image.jpg", "rb") as f:
    files = {"file": f}
    data = {"input_type": "bias_image"}

    response = requests.post(
        "http://localhost:8000/v1/analyze/upload",
        files=files,
        data=data
    )

result = response.json()
```

### WebSocket Streaming (Real-Time Events)

```python
import websockets
import asyncio
import json
import requests

async def stream_analysis():
    # Start analysis
    response = requests.post(
        "http://localhost:8000/v1/analyze/start",
        json={"content": "Text to analyze..."}
    )
    run_id = response.json()["run_id"]

    # Connect to event stream
    uri = f"ws://localhost:8000/v1/stream/{run_id}"
    async with websockets.connect(uri) as websocket:
        async for message in websocket:
            event = json.loads(message)

            print(f"[{event['event']}] {event['context'].get('message', '')}")

            # Check for completion
            if event["event"] == "analysis_complete":
                result = event["context"]["result"]
                print(f"\n✅ Complete: {result['summary']}")
                break

asyncio.run(stream_analysis())
```

### Health Check

```python
import requests

response = requests.get("http://localhost:8000/v1/health")
print(response.json())  # {"status": "ok"}
```

---

## Best Practices

### 1. Use Refinement for Critical Applications

```python
# Production: Enable refinement for high-stakes use cases
result = engine.analyze_text(
    job_posting,
    enable_refinement=True,
    max_refinement_iterations=2
)

# Development/Testing: Disable for speed
result = engine.analyze_text(
    test_text,
    enable_refinement=False
)
```

### 2. Cache Results

```python
# Enable caching to avoid re-analyzing identical content
from fairsense_agentix.configs.settings import Settings

settings = Settings(
    cache_enabled=True,
    cache_ttl_seconds=3600  # 1 hour
)
engine = FairSense(settings=settings)
```

### 3. Handle Errors Gracefully

```python
from fairsense_agentix.api import FairSense

engine = FairSense()

try:
    result = engine.analyze_text(text)
except Exception as e:
    print(f"Analysis failed: {e}")
    # Fallback logic or error reporting
```

### 4. Monitor Execution Time

```python
result = engine.analyze_text(text)

# Check performance
exec_time = result.metadata.execution_time_seconds
print(f"Completed in {exec_time:.2f}s")

if exec_time > 60:
    print("⚠️  Slow analysis - consider disabling refinement")
```

### 5. Aggregate Results for Reporting

```python
# Process multiple documents
results = [engine.analyze_text(doc) for doc in documents]

# Generate summary report
total_biases = sum(len(r.bias_instances) for r in results)
avg_risk = sum(1 for r in results if r.risk_level == "high") / len(results)

print(f"Analysis Summary:")
print(f"  Documents: {len(results)}")
print(f"  Total Biases: {total_biases}")
print(f"  High Risk: {avg_risk * 100:.1f}%")
```

---

## Next Steps

- **[API Reference](api.md)** - Complete API documentation
- **[Server Guide](server.md)** - Running the web interface and REST API
- **[GitHub Examples](https://github.com/VectorInstitute/fairsense-AgentiX/tree/main/examples)** - Additional code samples

---

## Troubleshooting

### "Analysis taking too long"

**Cause:** Refinement loop running multiple iterations

**Solution:**
```python
# Reduce refinement iterations
result = engine.analyze_text(
    text,
    max_refinement_iterations=1
)

# Or disable entirely
result = engine.analyze_text(text, enable_refinement=False)
```

### "Model download failed"

**Cause:** Network issues or disk space

**Solution:**
1. Check internet connection
2. Verify disk space: `df -h ~/.cache/huggingface`
3. Clear cache if needed: `rm -rf ~/.cache/huggingface`

### "API rate limit exceeded"

**Cause:** Too many LLM API calls

**Solution:**
```python
# Enable caching to reduce API calls
settings = Settings(cache_enabled=True)
engine = FairSense(settings=settings)

# Or use a local model
settings = Settings(
    llm_provider="openai",
    llm_base_url="http://localhost:11434/v1"
)
```

---

**Need more help?** Check the [Getting Started](getting_started.md) guide or [open an issue](https://github.com/VectorInstitute/fairsense-AgentiX/issues).
