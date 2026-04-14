"""LLM-based evaluators for post-hoc quality checks.

Implementation is split across focused modules:
    - :mod:`.common` — shared :class:`EvaluationContext` dataclass
    - :mod:`.bias`   — LLM-based bias evaluator
    - :mod:`.risk`   — rule-based risk evaluator

All public names are re-exported here so existing imports of the form
``from fairsense_agentix.services.evaluator import ...`` continue to work.
"""

from fairsense_agentix.services.evaluator.bias import (
    BiasEvaluatorOutput,
    _build_plain_langchain_model,
    _extract_forced_score,
    _invoke_bias_evaluator_llm,
    _serialize_bias_analysis,
    _simulate_bias_evaluator,
    evaluate_bias_output,
)
from fairsense_agentix.services.evaluator.common import EvaluationContext
from fairsense_agentix.services.evaluator.risk import (
    RiskEvaluatorOutput,
    _check_duplicate_risks,
    _check_risk_coverage,
    _check_rmf_breadth,
    _check_score_sanity,
    _compute_risk_score,
    evaluate_risk_output,
)


__all__ = [
    # common
    "EvaluationContext",
    # bias
    "BiasEvaluatorOutput",
    "evaluate_bias_output",
    "_simulate_bias_evaluator",
    "_extract_forced_score",
    "_invoke_bias_evaluator_llm",
    "_serialize_bias_analysis",
    "_build_plain_langchain_model",
    # risk
    "RiskEvaluatorOutput",
    "evaluate_risk_output",
    "_check_rmf_breadth",
    "_check_duplicate_risks",
    "_check_score_sanity",
    "_check_risk_coverage",
    "_compute_risk_score",
]
