#!/usr/bin/env python3
"""Transform MIT AI Risk Repository data into FairSense-AgentiX format.

This script processes the MIT AI Risk Repository V3 CSV files and creates:
1. ai_risks.csv - Risk taxonomy with descriptions for embedding
2. ai_rmf.csv - NIST AI-RMF recommendations mapped to risks

The script intelligently maps risks to appropriate RMF functions based on:
- Domain taxonomy (discrimination, privacy, misinformation, etc.)
- Timing (pre-deployment vs post-deployment)
- Intent (intentional vs unintentional)
"""

import csv
import logging
import os
import sys
from pathlib import Path


# Intentionally do not import `fairsense_agentix` here: importing the package runs
# `__init__.py` and pulls heavy deps (e.g. transformers). This script only needs stdlib.

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Clean and normalize text content.

    - Remove excessive whitespace
    - Remove quotes and special characters that cause CSV issues
    - Normalize line breaks
    """
    if not text or text == "":
        return ""
    # Remove newlines and excessive whitespace
    text = " ".join(text.split())
    # Remove problematic quotes
    text = text.replace('"""', '"').replace('""', '"')
    # Limit length for practical embedding
    if len(text) > 1000:
        text = text[:997] + "..."
    return text.strip()


def extract_risk_name(category: str, subcategory: str, description: str) -> str:
    """Extract a concise risk name from available fields."""
    if subcategory and subcategory.strip():
        return subcategory.strip()[:100]
    if category and category.strip():
        return category.strip()[:100]
    # Extract first sentence from description
    sentences = description.split(".")
    if sentences:
        return sentences[0].strip()[:100]
    return "Unnamed Risk"


def map_domain_to_severity(domain: str) -> str:
    """Map domain taxonomy to severity level.

    Some domains are inherently higher risk than others.
    """
    domain_lower = domain.lower()

    # Critical/High risk domains
    if any(
        keyword in domain_lower
        for keyword in [
            "malicious",
            "weapon",
            "cyberattack",
            "mass harm",
            "ai system safety",
            "failure",
            "toxic content",
        ]
    ):
        return "HIGH"

    # Medium risk domains
    if any(
        keyword in domain_lower
        for keyword in [
            "discrimination",
            "privacy",
            "misinformation",
            "socioeconomic",
            "governance",
        ]
    ):
        return "MEDIUM"

    # Default to medium
    return "MEDIUM"


def map_domain_to_rmf_functions(
    domain: str,
    timing: str,
    intent: str,
) -> list[dict[str, str]]:
    """Map a risk's characteristics to appropriate NIST AI-RMF functions.

    Returns a list of RMF recommendations with function assignments.
    Each risk gets 2-4 recommendations across different functions.
    """
    domain_lower = domain.lower()
    recommendations = []

    # GOVERN recommendations (policy, oversight, accountability)
    govern_templates = {
        "discrimination": (
            "Establish governance policies for fairness testing and bias monitoring "
            "across all AI systems"
        ),
        "privacy": (
            "Implement data governance frameworks ensuring privacy-by-design "
            "principles and compliance with regulations"
        ),
        "misinformation": (
            "Create oversight mechanisms to verify information accuracy and "
            "establish content moderation policies"
        ),
        "malicious": (
            "Define security governance policies and incident response procedures "
            "for AI system threats"
        ),
        "socioeconomic": (
            "Establish governance structures for monitoring societal impacts and "
            "stakeholder engagement"
        ),
        "ai system": (
            "Create technical governance frameworks for AI safety, testing "
            "protocols, and risk management"
        ),
        "human-computer": (
            "Develop human oversight policies ensuring appropriate human control "
            "and intervention capabilities"
        ),
    }

    # MAP recommendations (risk identification and categorization)
    map_templates = {
        "discrimination": (
            "Conduct comprehensive bias audits across protected characteristics "
            "and use cases"
        ),
        "privacy": (
            "Map data flows and identify privacy risks throughout the AI system "
            "lifecycle"
        ),
        "misinformation": (
            "Identify potential sources of misinformation and map information pathways"
        ),
        "malicious": (
            "Perform threat modeling to identify attack surfaces and security "
            "vulnerabilities"
        ),
        "socioeconomic": (
            "Map potential socioeconomic impacts across different stakeholder groups"
        ),
        "ai system": (
            "Conduct technical risk assessments identifying failure modes and "
            "safety boundaries"
        ),
        "human-computer": (
            "Map human-AI interaction points and identify areas requiring human "
            "oversight"
        ),
    }

    # MEASURE recommendations (testing, benchmarking, monitoring)
    measure_templates = {
        "discrimination": (
            "Implement continuous bias metrics tracking across demographic groups "
            "and contexts"
        ),
        "privacy": (
            "Deploy privacy-preserving measurement techniques and conduct regular "
            "privacy audits"
        ),
        "misinformation": (
            "Establish accuracy benchmarks and implement fact-checking validation "
            "protocols"
        ),
        "malicious": (
            "Deploy security monitoring systems and conduct penetration testing "
            "regularly"
        ),
        "socioeconomic": (
            "Measure societal impacts using validated metrics and stakeholder "
            "feedback mechanisms"
        ),
        "ai system": (
            "Implement comprehensive testing suites for robustness, reliability, "
            "and safety"
        ),
        "human-computer": (
            "Measure user trust, reliance patterns, and human-AI collaboration "
            "effectiveness"
        ),
    }

    # MANAGE recommendations (incident response, mitigation, improvement)
    manage_templates = {
        "discrimination": (
            "Establish bias remediation procedures and fairness improvement "
            "feedback loops"
        ),
        "privacy": (
            "Implement privacy incident response plans and data protection "
            "mitigation strategies"
        ),
        "misinformation": (
            "Create content correction mechanisms and misinformation mitigation "
            "workflows"
        ),
        "malicious": (
            "Deploy security incident response and threat mitigation capabilities"
        ),
        "socioeconomic": (
            "Establish harm mitigation procedures and stakeholder support mechanisms"
        ),
        "ai system": (
            "Implement safety fallbacks, failure recovery procedures, and "
            "continuous improvement processes"
        ),
        "human-computer": (
            "Manage human oversight escalation and intervention procedures"
        ),
    }

    # Select appropriate templates based on domain
    for keyword, template in govern_templates.items():
        if keyword in domain_lower:
            recommendations.append(
                {"function": "GOVERN", "action": template, "priority": "HIGH"},
            )
            break

    for keyword, template in map_templates.items():
        if keyword in domain_lower:
            recommendations.append(
                {
                    "function": "MAP",
                    "action": template,
                    "priority": "HIGH" if timing == "1 - Pre-deployment" else "MEDIUM",
                },
            )
            break

    for keyword, template in measure_templates.items():
        if keyword in domain_lower:
            recommendations.append(
                {"function": "MEASURE", "action": template, "priority": "HIGH"},
            )
            break

    for keyword, template in manage_templates.items():
        if keyword in domain_lower:
            recommendations.append(
                {
                    "function": "MANAGE",
                    "action": template,
                    "priority": "MEDIUM" if intent == "2 - Unintentional" else "HIGH",
                },
            )
            break

    # Ensure we have at least 3 functions represented (for evaluator breadth check)
    if len(recommendations) < 3:
        # Add default recommendations
        if not any(r["function"] == "GOVERN" for r in recommendations):
            recommendations.append(
                {
                    "function": "GOVERN",
                    "action": (
                        "Establish AI risk management governance policies and "
                        "accountability structures"
                    ),
                    "priority": "HIGH",
                },
            )
        if not any(r["function"] == "MAP" for r in recommendations):
            recommendations.append(
                {
                    "function": "MAP",
                    "action": (
                        "Conduct comprehensive risk assessment and impact analysis"
                    ),
                    "priority": "MEDIUM",
                },
            )
        if not any(r["function"] == "MEASURE" for r in recommendations):
            recommendations.append(
                {
                    "function": "MEASURE",
                    "action": (
                        "Implement monitoring and measurement systems for risk "
                        "indicators"
                    ),
                    "priority": "MEDIUM",
                },
            )

    return recommendations


def transform_risks_data(input_path: Path, output_path: Path) -> dict[str, dict]:  # noqa: PLR0915
    """Transform MIT AI Risk Database into ai_risks.csv format.

    Returns a dictionary mapping risk_id -> risk metadata for RMF generation.
    """
    logger.info("Reading risks from: %s", input_path)

    risks = []
    risk_metadata = {}  # For RMF generation
    risk_id_counter = 1

    with open(input_path, "r", encoding="utf-8") as f:
        # Find the header row containing "Category level"
        header_found = False
        for line_num, line in enumerate(f):
            if "Category level" in line:
                logger.debug("Found header row at line %s", line_num + 1)
                # Rewind to start of file
                f.seek(0)
                # Skip all lines before the header
                for _ in range(line_num):
                    next(f)
                header_found = True
                break

        if not header_found:
            logger.error(
                "Could not find header row with 'Category level' column",
            )
            return {}

        # Now read CSV starting from the actual header row
        reader = csv.DictReader(f)

        # Debug: Print actual column names
        if reader.fieldnames:
            logger.debug(
                "CSV columns found (first 10): %s",
                reader.fieldnames[:10],
            )

        total_rows = 0
        filtered_by_category = 0
        filtered_by_description = 0

        for row in reader:
            total_rows += 1

            # Debug: Print first few rows
            if total_rows <= 5:
                cat_level = row.get("Category level", "")
                desc_len = len(row.get("Description", ""))
                logger.debug(
                    "Row %s: category_level=%r description_length=%s",
                    total_rows,
                    cat_level,
                    desc_len,
                )

            # Skip header rows and paper-level entries
            category_level = row.get("Category level", "").strip()
            if category_level not in ["Risk Category", "Risk Sub-Category"]:
                filtered_by_category += 1
                continue

            # Skip rows without meaningful description
            description = clean_text(row.get("Description", ""))
            if not description or len(description) < 20:
                filtered_by_description += 1
                continue

            # Extract fields
            risk_category = clean_text(row.get("Risk category", ""))
            risk_subcategory = clean_text(row.get("Risk subcategory", ""))
            domain = clean_text(row.get("Domain", ""))
            subdomain = clean_text(row.get("Sub-domain", ""))
            entity = row.get("Entity", "")
            intent = row.get("Intent", "")
            timing = row.get("Timing", "")

            # Create risk entry
            risk_id = f"RISK{risk_id_counter:04d}"
            risk_name = extract_risk_name(risk_category, risk_subcategory, description)
            severity = map_domain_to_severity(domain)

            # Use domain subdomain as category if available
            category = subdomain if subdomain else domain

            risks.append(
                {
                    "id": risk_id,
                    "risk_name": risk_name,
                    "description": description,
                    "severity": severity,
                    "category": category,
                },
            )

            # Store metadata for RMF generation
            risk_metadata[risk_id] = {
                "domain": domain,
                "subdomain": subdomain,
                "entity": entity,
                "intent": intent,
                "timing": timing,
                "category": risk_category,
            }

            risk_id_counter += 1

    logger.info("")
    logger.info("Filtering summary:")
    logger.info("  Total rows read: %s", total_rows)
    logger.info("  Filtered by category level: %s", filtered_by_category)
    logger.info("  Filtered by description length: %s", filtered_by_description)
    logger.info("  Final risk entries: %s", len(risks))

    logger.info("")
    logger.info("Writing %s risks to: %s", len(risks), output_path)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "risk_name", "description", "severity", "category"],
        )
        writer.writeheader()
        writer.writerows(risks)

    logger.info("Created ai_risks.csv with %s risk entries", len(risks))
    return risk_metadata


def generate_rmf_recommendations(
    risk_metadata: dict[str, dict],
    output_path: Path,
) -> None:
    """Generate NIST AI-RMF recommendations for each risk."""
    logger.info("Generating RMF recommendations...")

    recommendations = []
    rmf_id_counter = 1

    for risk_id, metadata in risk_metadata.items():
        # Generate 2-4 recommendations per risk across different RMF functions
        rmf_recs = map_domain_to_rmf_functions(
            metadata["domain"],
            metadata["timing"],
            metadata["intent"],
        )

        for rec in rmf_recs:
            recommendations.append(
                {
                    "id": f"RMF{rmf_id_counter:04d}",
                    "risk_id": risk_id,
                    "function": rec["function"],
                    "action": rec["action"],
                    "priority": rec["priority"],
                },
            )
            rmf_id_counter += 1

    # Write ai_rmf.csv
    logger.info(
        "Writing %s RMF recommendations to: %s",
        len(recommendations),
        output_path,
    )
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "risk_id", "function", "action", "priority"],
        )
        writer.writeheader()
        writer.writerows(recommendations)

    logger.info(
        "Created ai_rmf.csv with %s RMF recommendations",
        len(recommendations),
    )

    # Statistics
    function_counts: dict[str, int] = {}
    for rec in recommendations:
        func = rec["function"]
        function_counts[func] = function_counts.get(func, 0) + 1

    logger.info("")
    logger.info("RMF function distribution:")
    for func in ["GOVERN", "MAP", "MEASURE", "MANAGE"]:
        count = function_counts.get(func, 0)
        percentage = (count / len(recommendations) * 100) if recommendations else 0
        logger.info("  %s: %s (%.1f%%)", func, count, percentage)


def main() -> int:
    """Run main transformation pipeline."""
    level_name = os.environ.get("LOGLEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    # Root may already have handlers (e.g. from other tools); force our CLI format.
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
        force=True,
    )

    print("=" * 70)
    print("MIT AI Risk Repository → FairSense-AgentiX Data Transformation")
    print("=" * 70)
    print()

    # Paths
    raw_dir = Path(__file__).parent.parent / "data" / "raw"
    input_file = raw_dir / (
        "Copy of The AI Risk Repository V3_26_03_2025 (please create a copy) "
        "- AI Risk Database v3.csv"
    )

    risks_output = raw_dir / "ai_risks.csv"
    rmf_output = raw_dir / "ai_rmf.csv"

    # Check input file exists
    if not input_file.exists():
        logger.error("Input file not found: %s", input_file)
        logger.error(
            "Please ensure the MIT AI Risk Repository CSV is in data/raw/",
        )
        return 1

    # Transform risks data
    risk_metadata = transform_risks_data(input_file, risks_output)

    # Generate RMF recommendations
    generate_rmf_recommendations(risk_metadata, rmf_output)

    print()
    print("=" * 70)
    print("Transformation Complete!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Review the generated files:")
    print(f"   - {risks_output}")
    print(f"   - {rmf_output}")
    print()
    print("2. Build FAISS indexes:")
    print("   $ uv run python scripts/build_faiss_indexes.py")
    print()
    print("3. Test with the UI:")
    print("   $ cd ui && npm run dev")
    print("   $ uv run python fairsense_agentix/service_api/server.py")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
