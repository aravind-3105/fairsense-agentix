"""Text generation tool protocol interfaces."""

from typing import TYPE_CHECKING, Protocol, runtime_checkable


if TYPE_CHECKING:
    from pydantic import BaseModel


@runtime_checkable
class LLMTool(Protocol):
    """Protocol for large language model (LLM) tools.

    Generates text completions from prompts using various LLM providers
    (OpenAI, Anthropic, local models, etc.).

    Examples
    --------
    >>> llm = OpenAILLM(model="gpt-4")
    >>> result = llm.predict("Analyze this text for bias: ...", temperature=0.3)
    >>> print(result)
    '**Bias Analysis**: The text contains...'
    """

    def predict(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        timeout: int = 60,
    ) -> str:
        """Generate text completion from prompt.

        Parameters
        ----------
        prompt : str
            Input prompt for the LLM
        temperature : float, optional
            Sampling temperature (0.0 = deterministic, 1.0 = creative),
            by default 0.3
        max_tokens : int, optional
            Maximum tokens to generate, by default 2000
        timeout : int, optional
            Request timeout in seconds, by default 60

        Returns
        -------
        str
            Generated text completion

        Raises
        ------
        LLMError
            If prediction fails (timeout, rate limit, API error, etc.)
        """
        ...

    def get_token_count(self, text: str) -> int:
        """Estimate token count for text.

        Used for cost tracking and prompt sizing.

        Parameters
        ----------
        text : str
            Text to count tokens for

        Returns
        -------
        int
            Estimated number of tokens
        """
        ...


@runtime_checkable
class SummarizerTool(Protocol):
    """Protocol for text summarization tools.

    Condenses long text into shorter summaries. Often implemented using LLMs
    but can also use extractive summarization methods.

    Examples
    --------
    >>> summarizer = LLMSummarizer()
    >>> summary = summarizer.summarize(long_text, max_length=200)
    >>> print(summary)
    'Summary: The text discusses...'
    """

    def summarize(
        self,
        text: str,
        max_length: int = 200,
    ) -> str:
        """Condense text into shorter summary.

        Parameters
        ----------
        text : str
            Text to summarize
        max_length : int, optional
            Maximum summary length in characters, by default 200

        Returns
        -------
        str
            Condensed summary of input text

        Raises
        ------
        LLMError
            If summarization fails
        """
        ...


@runtime_checkable
class VLMTool(Protocol):
    """Protocol for Vision-Language Model (VLM) tools.

    Analyzes images with structured output using vision-language models
    (GPT-4o Vision, Claude Sonnet Vision, etc.). Supports Chain-of-Thought
    reasoning and Pydantic schema validation.

    Examples
    --------
    >>> vlm = UnifiedVLMTool(settings)
    >>> result = vlm.analyze_image(
    ...     image_bytes=image_data,
    ...     prompt="Analyze this image for bias",
    ...     response_model=BiasVisualAnalysisOutput,
    ... )
    >>> print(result.bias_analysis.overall_assessment)
    'Gender bias detected in leadership representation...'
    """

    def analyze_image(
        self,
        image_bytes: bytes,
        prompt: str,
        response_model: type["BaseModel"],
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> "BaseModel":
        """Analyze image with structured Pydantic output.

        Sends image + prompt to Vision-Language Model and returns
        structured output matching the provided Pydantic schema.

        Parameters
        ----------
        image_bytes : bytes
            Raw image data (PNG, JPEG, etc.)
        prompt : str
            Analysis prompt (e.g., bias detection instructions)
        response_model : type[BaseModel]
            Pydantic model class for structured output
        temperature : float, optional
            Sampling temperature (0.0 = deterministic, 1.0 = creative),
            by default 0.3
        max_tokens : int, optional
            Maximum tokens to generate, by default 2000

        Returns
        -------
        BaseModel
            Instance of response_model with populated fields

        Raises
        ------
        VLMError
            If analysis fails (invalid image, API error, timeout, etc.)

        Notes
        -----
        Both OpenAI (GPT-4o Vision) and Anthropic (Claude Sonnet Vision)
        are supported. Provider is determined by llm_provider setting.
        """
        ...
