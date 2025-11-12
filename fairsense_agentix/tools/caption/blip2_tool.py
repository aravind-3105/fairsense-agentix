"""BLIP-2 image captioning tool for GPU-accelerated inference.

This module implements image captioning using Salesforce's BLIP-2 model.
BLIP-2 provides state-of-the-art quality with fast inference on GPU.

Design Choices
--------------
1. **Why BLIP-2**: Best quality (121.6 CIDEr score), 32-token compression
2. **Why GPU-first**: 50-200ms on GPU vs 2-5s on CPU (10-40x speedup)
3. **Why OPT-2.7B variant**: Balanced quality/speed (vs flan-t5 or larger models)
4. **Preload option**: Control startup latency vs first-call latency

What This Enables
-----------------
- Production-grade caption quality for bias detection
- Fast inference on GPU servers (50-200ms)
- Efficient token usage (32 tokens vs 256+ for other models)
- Better understanding of image context (people, activities, setting)
"""

import hashlib
import io
import logging

import torch
from PIL import Image

from fairsense_agentix.tools.exceptions import CaptionError


logger = logging.getLogger(__name__)


class BLIP2CaptionTool:
    """BLIP-2 image captioning tool for GPU-accelerated inference.

    Uses Salesforce BLIP-2 model (2.7B params) for state-of-the-art image
    captions. Optimized for GPU usage with fast inference (50-200ms).

    Performance
    -----------
    - GPU: 50-200ms per image (excellent)
    - CPU: 2-5s per image (too slow, use BLIP instead)

    Quality
    -------
    - Excellent: State-of-the-art zero-shot captioning
    - CIDEr score: 121.6 (vs previous best 113.2)
    - Use case: Production bias detection requiring high accuracy

    Examples
    --------
    >>> captioner = BLIP2CaptionTool(use_gpu=True, preload=True)
    >>> caption = captioner.caption(image_bytes, max_length=100)
    >>> print(caption)
    'a diverse group of professionals collaborating at a conference table'

    Notes
    -----
    Requires transformers and torch: pip install transformers torch accelerate
    Model automatically downloaded from Hugging Face on first use (~5GB).
    GPU acceleration requires CUDA-compatible GPU (8GB+ VRAM recommended).
    """

    def __init__(self, use_gpu: bool = True, preload: bool = True) -> None:
        """Initialize BLIP-2 caption tool.

        Design Choices:
        - **Default use_gpu=True**: BLIP-2 designed for GPU (too slow on CPU)
        - **Preload option**: Choose between startup time vs first-call latency
        - **In-memory cache**: Instant results for repeated images
        - **OPT-2.7B model**: Balanced quality/speed (vs larger 6.7B model)

        What This Enables:
        - Fast first API call if preload=True (20s startup, 100ms calls)
        - Fast app startup if preload=False (0s startup, 20s first call)
        - Automatic CPU fallback if GPU requested but unavailable
        - Production-grade quality with acceptable resource requirements

        Why This Way:
        - BLIP-2 is 10-40x faster on GPU than CPU (worth the GPU requirement)
        - 2.7B model fits in 8GB VRAM (vs 6.7B needs 16GB)
        - Preload=True better for API servers (amortize load time)
        - In-memory cache sufficient (captions small, ~50-200 chars)

        Parameters
        ----------
        use_gpu : bool, optional
            Use GPU if available, by default True (strongly recommended)
        preload : bool, optional
            Load model at initialization (True) or on first use (False),
            by default True

        Notes
        -----
        GPU mode requires CUDA-compatible GPU with 8GB+ VRAM.
        Falls back to CPU if GPU requested but unavailable (will be slow).
        Model download is ~5GB on first use, then cached locally.
        """
        self._model = None  # Lazy loading
        self._processor = None
        self._use_gpu = use_gpu
        self._device = None  # Determined at load time
        self._cache: dict[str, str] = {}  # SHA256 → caption mapping

        if preload:
            self._load_model()
        else:
            logger.info("BLIP2CaptionTool initialized (lazy loading enabled)")

    def _load_model(self) -> None:
        """Lazy load BLIP-2 model and processor.

        Design Choices:
        - **Salesforce/blip2-opt-2.7b**: Balanced model (2.7B params)
        - **float16**: Half precision for speed (no quality loss in inference)
        - **eval() mode**: Disable dropout for deterministic inference
        - **Device detection**: Check actual CUDA availability

        What This Enables:
        - Excellent quality (121.6 CIDEr score, state-of-the-art)
        - Fast inference (50-200ms on GPU with float16)
        - Fits in 8GB VRAM (vs 16GB for larger models)
        - Deterministic outputs (eval mode)

        Why This Way:
        - OPT-2.7B is sweet spot (vs flan-t5: same quality, faster)
        - float16 cuts memory/computation in half (safe for inference)
        - eval() is standard PyTorch practice for inference
        - Better to warn + use CPU than crash if GPU unavailable

        Raises
        ------
        CaptionError
            If transformers not installed or model loading fails
        """
        try:
            from transformers import (  # noqa: PLC0415
                Blip2ForConditionalGeneration,
                Blip2Processor,
            )

            # Determine actual device (GPU might be unavailable)
            # Design Choice: Check CUDA + warn if mismatch
            # What This Enables: Graceful CPU fallback, clear warnings
            # Why This Way: Better UX than crashing if GPU missing
            if self._use_gpu and torch.cuda.is_available():
                self._device = "cuda"  # type: ignore[assignment]
            else:
                self._device = "cpu"  # type: ignore[assignment]
                if self._use_gpu:
                    logger.warning(
                        "GPU requested but unavailable, using CPU (will be slow ~2-5s per image)"
                    )

            # Choose model variant
            # Design Choice: OPT-2.7B for balanced quality/speed
            # What This Enables: 8GB VRAM requirement, good quality
            # Why This Way: Other options: flan-t5-xl (same), opt-6.7b (needs 16GB)
            model_name = "Salesforce/blip2-opt-2.7b"
            logger.info(f"Loading BLIP-2 model: {model_name} (device={self._device})")

            # Load processor (handles image preprocessing)
            # Design Choice: Separate processor for clean concerns
            # What This Enables: Standardized preprocessing
            # Why This Way: transformers best practice
            self._processor = Blip2Processor.from_pretrained(model_name)  # type: ignore[assignment]

            # Load model with float16 for speed
            # Design Choice: float16 on GPU, float32 on CPU
            # What This Enables: 2x faster inference, half memory usage
            # Why This Way: float16 safe for inference, cuts resources in half

            # Security fallback: Use safetensors for torch < 2.6 (CVE-2025-32434)
            # Design Choice: Automatic safetensors on old torch versions
            # What This Enables: Works on manylinux_2_27 without security vulnerability
            # Why This Way: Safetensors is faster (76x CPU, 2x GPU) and more secure
            torch_version = tuple(
                int(x) for x in torch.__version__.split("+")[0].split(".")
            )
            use_safetensors = torch_version < (2, 6, 0)

            if use_safetensors:
                logger.info(
                    f"Using safetensors format (torch {torch.__version__} < 2.6, "
                    "avoids CVE-2025-32434 and 76x faster loading)"
                )

            self._model = Blip2ForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
                use_safetensors=use_safetensors,
            )
            self._model.to(self._device)  # type: ignore[attr-defined]
            self._model.eval()  # type: ignore[attr-defined]  # Inference mode (disable dropout)

            logger.info(f"BLIP-2 model loaded (device={self._device})")

        except ImportError as e:
            msg = (
                "transformers not installed. Install with: "
                "pip install transformers torch accelerate"
            )
            raise CaptionError(msg) from e
        except Exception as e:
            msg = "Failed to load BLIP-2 model"
            raise CaptionError(
                msg, context={"device": self._device, "error": str(e)}
            ) from e

    def caption(self, image_bytes: bytes, max_length: int = 100) -> str:
        """Generate caption for image using BLIP-2.

        Design Choices:
        - **SHA256 cache key**: Content-addressed caching
        - **Check cache first**: Instant return if seen before
        - **Beam search (num_beams=5)**: Better quality than greedy
        - **torch.no_grad()**: Disable gradients for speed

        What This Enables:
        - Instant results for repeated images (0ms vs 50-200ms)
        - High quality captions (beam search)
        - Efficient memory (no_grad saves ~30%)
        - Deterministic caching (same bytes = same key)

        Why This Way:
        - SHA256 is standard for content addressing
        - Cache-first prevents unnecessary computation
        - Beam search standard for caption quality
        - no_grad() is PyTorch best practice

        Parameters
        ----------
        image_bytes : bytes
            Raw image data (PNG, JPEG, etc.)
        max_length : int, optional
            Maximum caption length in tokens, by default 100

        Returns
        -------
        str
            Natural language caption describing image content

        Raises
        ------
        CaptionError
            If image invalid, unsupported format, or captioning fails

        Examples
        --------
        >>> captioner = BLIP2CaptionTool(use_gpu=True)
        >>> caption = captioner.caption(image_bytes)
        >>> print(caption)
        'a diverse group of people working together in a modern office'

        >>> # Short captions
        >>> caption = captioner.caption(image_bytes, max_length=50)

        >>> # Cached result (instant)
        >>> caption2 = captioner.caption(image_bytes)  # Returns immediately
        """
        # Check cache first
        # Design Choice: SHA256 hash of raw bytes as cache key
        # What This Enables: Same image = same key, works if persisted later
        # Why This Way: Content-addressed caching is deterministic
        cache_key = hashlib.sha256(image_bytes).hexdigest()
        if cache_key in self._cache:
            logger.debug("Caption cache hit")
            return self._cache[cache_key]

        # Lazy load model if not loaded
        if self._model is None:
            self._load_model()

        try:
            # Convert bytes → PIL Image
            # Design Choice: PIL for transformers compatibility
            # What This Enables: Handles all image formats
            # Why This Way: transformers expects PIL images
            image = Image.open(io.BytesIO(image_bytes))

            # Preprocess image
            # Design Choice: Use processor for standardized preprocessing
            # What This Enables: Correct format (resize, normalize, tensor)
            # Why This Way: Processor handles all model requirements
            inputs = self._processor(image, return_tensors="pt").to(self._device)  # type: ignore[misc]

            # Generate caption with beam search
            # Design Choice: num_beams=5 for quality, max_length for control
            # What This Enables: Better captions than greedy decoding
            # Why This Way: Beam search is standard for seq2seq quality
            with torch.no_grad():  # Disable gradients for inference
                output = self._model.generate(  # type: ignore[attr-defined]
                    **inputs,
                    max_length=max_length,
                    num_beams=5,  # Beam search for better quality
                )

            # Decode output tokens → text
            # Design Choice: skip_special_tokens=True removes internal tokens
            # What This Enables: Clean caption text
            # Why This Way: Users don't need to see model internals
            caption = self._processor.decode(output[0], skip_special_tokens=True)  # type: ignore[attr-defined]

            # Cache result
            # Design Choice: Store in memory for instant future lookups
            # What This Enables: 0ms retrieval for repeated images
            # Why This Way: Captions are tiny (~50-200 chars)
            self._cache[cache_key] = caption

            logger.debug(
                f"Generated caption ({len(caption)} chars, device={self._device})"
            )
            return caption

        except OSError as e:
            # Image format issues
            msg = "Invalid or corrupted image"
            raise CaptionError(
                msg,
                context={
                    "image_size_bytes": len(image_bytes),
                    "device": self._device,
                    "error": str(e),
                },
            ) from e
        except Exception as e:
            # Catch-all for unexpected errors
            msg = "Caption generation failed"
            raise CaptionError(
                msg,
                context={
                    "image_size_bytes": len(image_bytes),
                    "device": self._device,
                    "max_length": max_length,
                    "error": str(e),
                },
            ) from e
