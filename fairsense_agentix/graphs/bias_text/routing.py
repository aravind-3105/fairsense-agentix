"""Conditional edges for the bias text graph."""

from fairsense_agentix.graphs.state import BiasTextState


def should_summarize(state: BiasTextState) -> bool:
    """Conditional edge: Decide whether to generate summary.

    Summarization is triggered if:
    - Text is long (>500 chars)
    - User explicitly requested it via options

    Parameters
    ----------
    state : BiasTextState
        Current state

    Returns
    -------
    bool
        True if summary should be generated
    """
    # Check text length
    if len(state.text) > 500:
        return True

    # Check options
    options = state.options or {}
    return bool(options.get("enable_summary", False))
