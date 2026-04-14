"""Rule-based evaluator for post-hoc quality checks on risk workflow output."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from fairsense_agentix.configs import settings
from fairsense_agentix.graphs.state import EvaluationResult
from fairsense_agentix.services.telemetry import telemetry

from .common import EvaluationContext


logger = logging.getLogger(__name__)


class RiskEvaluatorOutput(BaseModel):
    """Structured response from the risk evaluator.

    Phase 7.2: Rule-based quality checks for risk assessment output.
    Unlike bias evaluator (LLM-based), risk evaluator uses statistical checks.
    """

    score: int = Field(..., ge=0, le=100, description="Quality score (0-100)")
    justification: str = Field(..., description="Why this score was assigned")
    issues: list[str] = Field(
        default_factory=list,
        description="Specific quality issues found",
    )
    suggested_changes: dict[str, Any] = Field(
        default_factory=dict,
        description="Refinement hints (e.g., {'top_k': 10})",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Diagnostic info (RMF function counts, duplicate count, etc.)",
    )


def evaluate_risk_output(
    workflow_result: dict[str, Any],
    *,
    options: dict[str, Any],
    context: EvaluationContext,
) -> EvaluationResult:
    """Evaluate risk workflow output for quality.

    Phase 7.2: Rule-based quality checks for risk assessment results.
    Unlike bias evaluator (LLM-based), risk evaluator uses statistical checks.

    Quality Checks:
    - RMF function breadth (≥ 3 different functions)
    - Duplicate risk detection
    - FAISS score sanity (avg > threshold)
    - Sufficient risk coverage

    Parameters
    ----------
    workflow_result : dict[str, Any]
        Output from risk graph (risks, rmf_recommendations, joined_table)
    options : dict[str, Any]
        Orchestrator options (may contain previous refinement hints)
    context : EvaluationContext
        Metadata (workflow_id, run_id)

    Returns
    -------
    EvaluationResult
        Evaluation with pass/fail, score, issues, refinement hints

    Examples
    --------
    >>> result = {"risks": [...], "rmf_recommendations": {...}}
    >>> evaluation = evaluate_risk_output(result, options={}, context=...)
    >>> evaluation.passed
    True
    """
    if not settings.evaluator_enabled:
        logger.debug("Risk evaluator disabled; returning pass-through result")
        return EvaluationResult(
            passed=True,
            score=1.0,
            issues=[],
            explanation="Risk evaluators disabled",
        )

    telemetry.log_info(
        "risk_evaluator_start",
        workflow_id=context.workflow_id,
        run_id=context.run_id,
    )

    risks = workflow_result.get("risks", [])
    rmf_recommendations = workflow_result.get("rmf_recommendations", {})

    if not risks:
        logger.warning("Risk evaluator: No risks found in workflow result")
        return EvaluationResult(
            passed=False,
            score=0.0,
            issues=["No risks found in assessment"],
            refinement_hints={
                "options": {
                    "top_k": options.get("top_k", 5) + 5,  # K-bump
                },
            },
            explanation="Risk assessment returned no results",
            metadata={"risk": {"risk_count": 0, "empty_result": True}},
        )

    breadth_passed, function_count, breadth_reason = _check_rmf_breadth(
        rmf_recommendations,
        min_functions=settings.risk_evaluator_min_rmf_functions,
    )

    duplicates_passed, dup_count, dup_pairs = _check_duplicate_risks(
        risks,
        similarity_threshold=settings.risk_duplicate_threshold,
    )

    score_passed, avg_score, score_reason = _check_score_sanity(
        risks,
        min_avg_score=settings.risk_evaluator_min_avg_score,
    )

    coverage_passed, risk_count, coverage_reason = _check_risk_coverage(
        risks,
        min_risks=settings.risk_evaluator_min_risks,
    )

    all_passed = (
        breadth_passed and duplicates_passed and score_passed and coverage_passed
    )

    issues = []
    if not breadth_passed:
        issues.append(f"RMF breadth insufficient: {breadth_reason}")
    if not duplicates_passed:
        issues.append(f"Duplicate risks found: {dup_count} duplicate pairs")
    if not score_passed:
        issues.append(f"Low FAISS scores: {score_reason}")
    if not coverage_passed:
        issues.append(f"Insufficient coverage: {coverage_reason}")

    score = _compute_risk_score(
        breadth_passed,
        duplicates_passed,
        score_passed,
        coverage_passed,
    )

    refinement_hints: dict[str, Any] = {}
    if not all_passed:
        current_top_k = options.get("top_k", 5)
        refinement_hints = {
            "options": {
                "top_k": min(current_top_k + 5, 20),  # K-bump with cap at 20
            },
        }

    metadata = {
        "risk": {
            "score": score,
            "function_count": function_count,
            "duplicate_count": dup_count,
            "avg_faiss_score": avg_score,
            "risk_count": risk_count,
            "checks": {
                "rmf_breadth": breadth_passed,
                "duplicates": duplicates_passed,
                "score_sanity": score_passed,
                "coverage": coverage_passed,
            },
            "workflow_id": context.workflow_id,
            "run_id": context.run_id,
        },
    }

    explanation = (
        f"Risk evaluation: {score}/100 - {len(issues)} issue(s) found"
        if issues
        else "All quality checks passed"
    )

    telemetry.log_info(
        "risk_evaluator_complete",
        passed=all_passed,
        score=score,
        issues_count=len(issues),
    )

    return EvaluationResult(
        passed=all_passed,
        score=score / 100,
        issues=issues,
        refinement_hints=refinement_hints,
        explanation=explanation,
        metadata=metadata,
    )


def _check_rmf_breadth(
    rmf_recommendations: dict[str, list[dict[str, Any]]],
    min_functions: int = 3,
) -> tuple[bool, int, str]:
    """Check if RMF recommendations cover diverse functions.

    AI-RMF has 4 core functions: Govern, Map, Measure, Manage.
    Good coverage means recommendations span ≥ min_functions.

    Parameters
    ----------
    rmf_recommendations : dict
        RMF recommendations keyed by risk_id
    min_functions : int
        Minimum number of unique RMF functions required

    Returns
    -------
    tuple[bool, int, str]
        (passed, function_count, justification)

    Examples
    --------
    >>> rmf = {"RISK1": [{"function": "Govern"}, {"function": "Measure"}]}
    >>> passed, count, reason = _check_rmf_breadth(rmf, min_functions=2)
    >>> passed
    True
    """
    if not rmf_recommendations:
        return False, 0, "No RMF recommendations found"

    functions_found = set()
    for _risk_id, recs in rmf_recommendations.items():
        for rec in recs:
            func = rec.get("function") or rec.get("rmf_function", "")
            if func:
                functions_found.add(func.upper())

    function_count = len(functions_found)
    passed = function_count >= min_functions

    if passed:
        justification = (
            f"Good RMF breadth: {function_count} unique functions "
            f"({', '.join(sorted(functions_found))})"
        )
    else:
        justification = (
            f"Only {function_count}/{min_functions} RMF functions found "
            f"({', '.join(sorted(functions_found)) if functions_found else 'none'})"
        )

    return passed, function_count, justification


def _check_duplicate_risks(
    risks: list[dict[str, Any]],
    similarity_threshold: float = 0.9,
) -> tuple[bool, int, list[tuple[str, str]]]:
    """Check for near-duplicate risk descriptions.

    Uses simple string comparison (can be enhanced with embeddings later).

    Parameters
    ----------
    risks : list[dict]
        List of risk dictionaries with descriptions
    similarity_threshold : float
        Similarity threshold for duplicate detection (0.0-1.0)

    Returns
    -------
    tuple[bool, int, list[tuple[str, str]]]
        (passed, duplicate_count, [(risk1_id, risk2_id)])

    Examples
    --------
    >>> risks = [
    ...     {"id": "R1", "description": "Bias in AI"},
    ...     {"id": "R2", "description": "Bias in AI"},
    ... ]
    >>> passed, count, pairs = _check_duplicate_risks(risks)
    >>> passed
    False
    """
    if len(risks) < 2:
        return True, 0, []

    duplicate_pairs = []

    for i in range(len(risks)):
        for j in range(i + 1, len(risks)):
            desc1 = risks[i].get("description") or risks[i].get("text", "")
            desc2 = risks[j].get("description") or risks[j].get("text", "")

            desc1_norm = desc1.lower().strip()
            desc2_norm = desc2.lower().strip()

            if desc1_norm and desc2_norm and desc1_norm == desc2_norm:
                risk1_id = risks[i].get("id") or risks[i].get("risk_id", f"risk_{i}")
                risk2_id = risks[j].get("id") or risks[j].get("risk_id", f"risk_{j}")
                duplicate_pairs.append((risk1_id, risk2_id))

    duplicate_count = len(duplicate_pairs)
    passed = duplicate_count == 0

    return passed, duplicate_count, duplicate_pairs


def _check_score_sanity(
    risks: list[dict[str, Any]],
    min_avg_score: float = 0.3,
) -> tuple[bool, float, str]:
    """Check if FAISS similarity scores are reasonable.

    Parameters
    ----------
    risks : list[dict]
        List of risk dictionaries with similarity scores
    min_avg_score : float
        Minimum average score threshold (0.0-1.0)

    Returns
    -------
    tuple[bool, float, str]
        (passed, avg_score, justification)

    Examples
    --------
    >>> risks = [{"score": 0.8}, {"score": 0.7}, {"score": 0.6}]
    >>> passed, avg, reason = _check_score_sanity(risks, min_avg_score=0.5)
    >>> passed
    True
    """
    if not risks:
        return False, 0.0, "No risks to evaluate"

    scores = []
    for risk in risks:
        score = risk.get("score") or risk.get("risk_score", 0.0)
        # Handle both similarity score (0-1) and percentage (0-100)
        if score > 1.0:
            score = score / 100.0
        scores.append(score)

    if not scores:
        return False, 0.0, "No valid scores found in risks"

    avg_score = sum(scores) / len(scores)
    passed = avg_score >= min_avg_score

    if passed:
        justification = (
            f"Good FAISS scores: avg={avg_score:.2f} "
            f"(min={min(scores):.2f}, max={max(scores):.2f})"
        )
    else:
        justification = (
            f"Low FAISS scores: avg={avg_score:.2f} < {min_avg_score:.2f} "
            f"(indicates weak relevance)"
        )

    return passed, avg_score, justification


def _check_risk_coverage(
    risks: list[dict[str, Any]],
    min_risks: int = 3,
) -> tuple[bool, int, str]:
    """Check if sufficient risks were retrieved.

    Parameters
    ----------
    risks : list[dict]
        List of risk dictionaries
    min_risks : int
        Minimum number of risks required

    Returns
    -------
    tuple[bool, int, str]
        (passed, risk_count, justification)

    Examples
    --------
    >>> risks = [{"id": "R1"}, {"id": "R2"}, {"id": "R3"}]
    >>> passed, count, reason = _check_risk_coverage(risks, min_risks=2)
    >>> passed
    True
    """
    risk_count = len(risks)
    passed = risk_count >= min_risks

    if passed:
        justification = f"Sufficient coverage: {risk_count} risks found"
    else:
        justification = (
            f"Insufficient coverage: only {risk_count}/{min_risks} risks found"
        )

    return passed, risk_count, justification


def _compute_risk_score(
    breadth_passed: bool,
    duplicates_passed: bool,
    score_passed: bool,
    coverage_passed: bool,
) -> int:
    """Compute overall risk evaluation score (0-100).

    Each check contributes 25 points to the total score.

    Parameters
    ----------
    breadth_passed : bool
        RMF breadth check passed
    duplicates_passed : bool
        Duplicate check passed
    score_passed : bool
        Score sanity check passed
    coverage_passed : bool
        Coverage check passed

    Returns
    -------
    int
        Score between 0-100

    Examples
    --------
    >>> score = _compute_risk_score(True, True, True, False)
    >>> score
    75
    """
    checks = [breadth_passed, duplicates_passed, score_passed, coverage_passed]
    passed_count = sum(checks)
    return passed_count * 25
