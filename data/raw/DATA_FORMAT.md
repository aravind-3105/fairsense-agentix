# Data Format Specification

This directory contains raw CSV files used to build FAISS indexes for AI risk assessment.

## Required Files

### 1. ai_risks.csv
AI risk taxonomy with the following columns:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `id` | string | Unique risk identifier | "RISK001" |
| `risk_name` | string | Short name for the risk | "Algorithmic Bias" |
| `description` | string | Detailed description (used for embedding) | "Systematic errors in AI decisions..." |
| `severity` | string | Risk severity level | "HIGH", "MEDIUM", "LOW" |
| `category` | string (optional) | Risk category | "Fairness", "Safety", "Privacy" |

**Example:**
```csv
id,risk_name,description,severity,category
RISK001,Algorithmic Bias,"Systematic and repeatable errors in AI systems that create unfair outcomes for certain groups",HIGH,Fairness
RISK002,Data Privacy Violation,"Unauthorized collection or disclosure of personal information",CRITICAL,Privacy
```

### 2. ai_rmf.csv
NIST AI Risk Management Framework recommendations with the following columns:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `id` | string | Unique recommendation ID | "RMF001" |
| `risk_id` | string | Associated risk ID (foreign key) | "RISK001" |
| `function` | string | RMF function | "GOVERN", "MAP", "MEASURE", "MANAGE" |
| `action` | string | Recommended action (used for embedding) | "Implement bias testing protocols..." |
| `priority` | string (optional) | Implementation priority | "HIGH", "MEDIUM", "LOW" |

**Example:**
```csv
id,risk_id,function,action,priority
RMF001,RISK001,MEASURE,"Implement statistical tests to detect bias across demographic groups before deployment",HIGH
RMF002,RISK001,MANAGE,"Establish regular auditing procedures to monitor for bias in production",MEDIUM
```

## Building Indexes

After providing these CSV files, run:
```bash
uv run python scripts/build_faiss_indexes.py
```

This will generate:
- `data/indexes/risks.faiss` - FAISS index for risk descriptions
- `data/indexes/risks_meta.json` - Metadata (IDs, original text)
- `data/indexes/rmf.faiss` - FAISS index for RMF actions
- `data/indexes/rmf_meta.json` - Metadata for recommendations

## Notes

- The `description` field in ai_risks.csv is what gets embedded for similarity search
- The `action` field in ai_rmf.csv is what gets embedded
- Make sure text fields have substantial content (>20 words) for meaningful embeddings
- UTF-8 encoding required
