"""LLM-based evaluators for post-hoc quality checks."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, cast

from langchain_anthropic import ChatAnthropic
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr

from fairsense_agentix.configs import settings
from fairsense_agentix.graphs.state import EvaluationResult
from fairsense_agentix.prompts.prompt_loader import PromptLoader
from fairsense_agentix.services.telemetry import telemetry
from fairsense_agentix.tools.llm.callbacks import TelemetryCallback
from fairsense_agentix.tools.llm.output_schemas import BiasAnalysisOutput


logger = logging.getLogger(__name__)


@dataclass
class EvaluationContext:
    """Container describing what was evaluated (for metadata)."""

    workflow_id: str
    run_id: str | None


class BiasEvaluatorOutput(BaseModel):
    """Structured response from the bias evaluator critique."""

    score: int = Field(..., ge=0, le=100)
    justification: str = Field(..., description="Why this score was assigned")
    suggested_changes: list[str] = Field(
        default_factory=list,
        description="Concrete actions the next run should apply",
    )


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


def evaluate_bias_output(
    workflow_result: dict[str, Any],
    *,
    original_text: str | None,
    options: dict[str, Any],
    context: EvaluationContext,
) -> EvaluationResult:
    """Evaluate bias workflow output with a single-shot critique.

    Parameters
    ----------
    workflow_result : dict[str, Any]
        Output payload from bias_text or bias_image workflow.
    original_text : str | None
        Source text (for text workflow) or merged OCR+caption text for image workflow.
    options : dict[str, Any]
        Orchestrator options (used for debug hooks / refinement hints).
    context : EvaluationContext
        Metadata describing the workflow under evaluation.
    """
    if not settings.evaluator_enabled:
        logger.debug("Evaluator disabled; returning pass-through result")
        return EvaluationResult(
            passed=True,
            score=1.0,
            issues=[],
            explanation="Evaluators disabled",
        )

    # Allow tests and debug runs to force a score before calling the LLM.
    forced_score = _extract_forced_score(options)
    feedback_applied = bool(options.get("bias_prompt_feedback"))

    if forced_score is not None and not feedback_applied:
        critique = BiasEvaluatorOutput(
            score=forced_score,
            justification="Forced score for testing",
            suggested_changes=[
                "Address evaluator feedback by covering missing categories",
                "Ground bias spans explicitly in the source text",
            ],
        )
    elif settings.llm_provider == "fake":
        critique = _simulate_bias_evaluator(feedback_applied)
    else:
        critique = _invoke_bias_evaluator_llm(
            workflow_result=workflow_result,
            original_text=original_text,
            existing_feedback=options.get("bias_prompt_feedback") or [],
        )

    passed = critique.score >= settings.bias_evaluator_min_score
    metadata = {
        "bias": {
            "score": critique.score,
            "threshold": settings.bias_evaluator_min_score,
            "justification": critique.justification,
            "suggested_changes": critique.suggested_changes,
            "workflow_id": context.workflow_id,
            "run_id": context.run_id,
        },
    }

    refinement_hints: dict[str, Any] = {}
    issues: list[str] = []

    if not passed:
        issues.append(critique.justification)
        feedback = critique.suggested_changes or [
            "Re-check every bias category and cite exact spans.",
        ]
        refinement_hints = {
            "options": {
                "bias_prompt_feedback": feedback,
            },
        }

    return EvaluationResult(
        passed=passed,
        score=critique.score / 100,
        issues=issues,
        refinement_hints=refinement_hints,
        explanation=critique.justification,
        metadata=metadata,
    )


def _simulate_bias_evaluator(feedback_applied: bool) -> BiasEvaluatorOutput:
    """Deterministic evaluator used when llm_provider=fake."""
    if feedback_applied:
        return BiasEvaluatorOutput(
            score=95,
            justification="Feedback applied; auto-accepting bias analysis.",
            suggested_changes=[],
        )

    return BiasEvaluatorOutput(
        score=85,
        justification=(
            "Simulation mode (fake provider): analysis considered sufficient."
        ),
        suggested_changes=[],
    )


def _extract_forced_score(options: dict[str, Any]) -> int | None:
    """Read the debug hook for forcing evaluator score (testing only)."""
    raw_value = options.get("force_bias_eval_score")
    if raw_value is None:
        return None
    try:
        numeric = float(raw_value)
    except (TypeError, ValueError):
        logger.warning("force_bias_eval_score must be numeric; ignoring %r", raw_value)
        return None

    # Support 0-1 or 0-100 ranges.
    if 0 <= numeric <= 1:
        numeric *= 100

    numeric = max(0, min(100, int(numeric)))
    logger.debug("Forcing bias evaluator score to %s", numeric)
    return numeric


def _invoke_bias_evaluator_llm(
    *,
    workflow_result: dict[str, Any],
    original_text: str | None,
    existing_feedback: list[str],
) -> BiasEvaluatorOutput:
    """Call LangChain LLM with structured output for the critique."""
    prompt_loader = PromptLoader()
    prompt_template = prompt_loader.load("bias_evaluator_v1.txt")
    analysis_section = _serialize_bias_analysis(workflow_result.get("bias_analysis"))
    summary_section = workflow_result.get("summary") or ""
    merged_text = original_text or workflow_result.get("merged_text") or ""
    existing_feedback_text = (
        "\n".join(f"- {item}" for item in existing_feedback)
        if existing_feedback
        else "None"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You critique fairness analyses and return structured "
                    "JSON as instructed."
                ),
            ),
            (
                "user",
                prompt_template,
            ),
        ],
    ).partial(
        bias_analysis_json=analysis_section,
        analysis_summary=summary_section or "None",
        source_text=merged_text or "Not available",
        existing_feedback=existing_feedback_text,
    )

    parser = PydanticOutputParser(pydantic_object=BiasEvaluatorOutput)
    langchain_model = _build_plain_langchain_model()
    chain = prompt | langchain_model | parser
    logger.debug("Invoking bias evaluator LLM")
    return chain.invoke({})


def _serialize_bias_analysis(analysis: Any) -> str:
    """Convert bias analysis payload to readable JSON for evaluator prompt."""
    if isinstance(analysis, BiasAnalysisOutput):
        return json.dumps(
            analysis.model_dump(),
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    if isinstance(analysis, str):
        # Already serialized (fake provider)
        return analysis
    if analysis is None:
        return "null"
    return json.dumps(analysis, indent=2, default=str, ensure_ascii=False)


def _build_plain_langchain_model() -> BaseLanguageModel:
    """Create a plain LangChain chat model with telemetry + caching."""
    provider = settings.llm_provider
    callback = TelemetryCallback(
        telemetry=telemetry,
        provider=provider,
        model=settings.llm_model_name,
    )

    if settings.llm_cache_enabled:
        settings.llm_cache_path.parent.mkdir(parents=True, exist_ok=True)
        set_llm_cache(SQLiteCache(database_path=str(settings.llm_cache_path)))

    if provider == "openai":
        base_model = ChatOpenAI(
            model=settings.llm_model_name,
            api_key=SecretStr(settings.llm_api_key) if settings.llm_api_key else None,
            callbacks=[callback],
        )
        return cast(BaseLanguageModel, base_model.with_retry(stop_after_attempt=3))

    if provider == "anthropic":
        if not settings.llm_api_key:
            raise ValueError("Anthropic provider requires API key for evaluator.")

        anthropic_model = ChatAnthropic(  # type: ignore[call-arg]
            model_name=settings.llm_model_name,
            api_key=SecretStr(settings.llm_api_key),
            callbacks=[callback],
        )
        return cast(BaseLanguageModel, anthropic_model.with_retry(stop_after_attempt=3))

    raise ValueError(f"Unsupported LLM provider for evaluator: {provider}")


# ============================================================================
# Phase 7.2: Risk Evaluator
# ============================================================================


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

    # Extract data from workflow_result
    risks = workflow_result.get("risks", [])
    rmf_recommendations = workflow_result.get("rmf_recommendations", {})

    # Handle empty results early
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

    # Run all quality checks
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

    # Aggregate results
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

    # Compute score (0-100) based on checks passed
    score = _compute_risk_score(
        breadth_passed,
        duplicates_passed,
        score_passed,
        coverage_passed,
    )

    # Generate refinement hints if failing
    refinement_hints: dict[str, Any] = {}
    if not all_passed:
        current_top_k = options.get("top_k", 5)
        refinement_hints = {
            "options": {
                "top_k": min(current_top_k + 5, 20),  # K-bump with cap at 20
            },
        }

    # Build metadata
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
        score=score / 100,  # Convert to 0-1 scale
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

    # Extract all RMF functions from recommendations
    functions_found = set()
    for _risk_id, recs in rmf_recommendations.items():
        for rec in recs:
            func = rec.get("function") or rec.get("rmf_function", "")
            if func:
                # Normalize to uppercase (handle "Govern" vs "GOVERN")
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
        return True, 0, []  # Can't have duplicates with < 2 risks

    duplicate_pairs = []

    # Simple string-based duplicate detection
    # Future enhancement: Use embeddings for semantic similarity
    for i in range(len(risks)):
        for j in range(i + 1, len(risks)):
            desc1 = risks[i].get("description") or risks[i].get("text", "")
            desc2 = risks[j].get("description") or risks[j].get("text", "")

            # Normalize for comparison
            desc1_norm = desc1.lower().strip()
            desc2_norm = desc2.lower().strip()

            # Check for exact or very similar matches (combined condition)
            if desc1_norm and desc2_norm and desc1_norm == desc2_norm:
                # Exact duplicate
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

    # Extract scores
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
    # Each check worth 25 points
    return passed_count * 25
