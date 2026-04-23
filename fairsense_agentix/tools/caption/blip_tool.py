"""BLIP image captioning tool for CPU-friendly inference.

This module implements image captioning using Salesforce's BLIP (v1) model.
BLIP provides good quality captions with fast inference on CPU.

Design Choices
--------------
1. **Why BLIP (v1)**: Smaller model (476M params), 2-3x faster than BLIP-2 on CPU
2. **Why preload option**: Choose startup time vs first-call latency trade-off
3. **Why in-memory cache**: Fast lookups (0ms) for repeated images
4. **Lazy loading**: Only import transformers when first needed

What This Enables
-----------------
- Acceptable speed on CPU (500ms-1s per image)
- Lower RAM requirements (vs BLIP-2: 2GB vs 8GB)
- Works on laptops without GPU
- Good enough quality for most bias detection use cases
"""

import hashlib
import io
import logging

import torch
from PIL import Image

from fairsense_agentix.tools.exceptions import CaptionError


logger = logging.getLogger(__name__)


class BLIPCaptionTool:
    """BLIP image captioning tool for CPU-friendly inference.

    Uses Salesforce BLIP model (476M params) for fast, good-quality image
    captions. Optimized for CPU usage with acceptable speed (500ms-1s).

    Performance
    -----------
    - CPU: 500ms-1s per image (acceptable)
    - GPU: 100-200ms per image (good)

    Quality
    -------
    - Good: Captures main objects and actions
    - Use case: Sufficient for most bias detection scenarios

    Examples
    --------
    >>> captioner = BLIPCaptionTool(use_gpu=False, preload=True)
    >>> caption = captioner.caption(image_bytes, max_length=100)
    >>> print(caption)
    'a group of people sitting at a desk with laptops'

    Notes
    -----
    Requires transformers and torch: pip install transformers torch
    Model automatically downloaded from Hugging Face on first use (~1GB).
    """

    def __init__(self, use_gpu: bool = False, preload: bool = True) -> None:
        """Initialize BLIP caption tool.

        Design Choices:
        - **Lazy loading**: Defer model import until needed (saves startup time)
        - **Preload option**: Choose between fast startup vs fast first call
        - **In-memory cache**: Dict for instant cache hits (no persistence needed)
        - **Device auto-detection**: Respect use_gpu but check CUDA availability

        What This Enables:
        - Fast app startup if preload=False (defer 5-10s load time)
        - Fast first API call if preload=True (load at initialization)
        - Instant results for repeated images (in-memory cache)
        - Automatic CPU fallback if GPU requested but unavailable

        Why This Way:
        - transformers import takes 2-3s (large package)
        - Model download is 1GB (first time only, then cached)
        - Preload choice gives users control over latency trade-offs
        - In-memory cache sufficient (captions are small, ~50-200 chars)

        Parameters
        ----------
        use_gpu : bool, optional
            Use GPU if available, by default False (CPU-optimized)
        preload : bool, optional
            Load model at initialization (True) or on first use (False),
            by default True

        Notes
        -----
        GPU mode requires CUDA-compatible GPU and drivers.
        Falls back to CPU if GPU requested but unavailable.
        """
        self._model = None  # Lazy loading
        self._processor = None
        self._use_gpu = use_gpu
        self._device = None  # Determined at load time
        self._cache: dict[str, str] = {}  # SHA256 → caption mapping

        if preload:
            self._load_model()
        else:
            logger.info("BLIPCaptionTool initialized (lazy loading enabled)")

    def _load_model(self) -> None:
        """Lazy load BLIP model and processor.

        Design Choices:
        - **Salesforce/blip-image-captioning-base**: Balanced model (476M params)
        - **float16 on GPU**: Half precision for speed (no quality loss)
        - **eval() mode**: Disable dropout for deterministic inference
        - **Device detection**: Check actual CUDA availability

        What This Enables:
        - Fast downloads (1GB vs 5GB for BLIP-2)
        - Efficient GPU memory (float16 uses half the RAM)
        - Deterministic outputs (eval mode disables randomness)
        - Graceful CPU fallback if GPU unavailable

        Why This Way:
        - blip-base is the standard, well-tested model
        - float16 is safe for inference (training needs float32)
        - eval() is standard PyTorch practice for inference
        - Better to run on CPU than crash if GPU unavailable

        Raises
        ------
        CaptionError
            If transformers not installed or model loading fails
        """
        import time

        load_start = time.time()
        logger.info("⏱️  [BLIP] Starting model load...")

        try:
            from transformers import (  # noqa: PLC0415
                BlipForConditionalGeneration,
                BlipProcessor,
            )

            # Determine actual device (GPU might be unavailable)
            # Design Choice: Check CUDA availability, not just user preference
            # What This Enables: Automatic CPU fallback, no crashes
            # Why This Way: Better UX than crashing if GPU requested but missing
            if self._use_gpu and torch.cuda.is_available():
                self._device = "cuda"  # type: ignore[assignment]
            else:
                self._device = "cpu"  # type: ignore[assignment]
                if self._use_gpu:
                    logger.warning("GPU requested but unavailable, using CPU")

            model_name = "Salesforce/blip-image-captioning-base"
            logger.info(
                f"⏱️  [BLIP] Loading model: {model_name} (device={self._device})",
            )

            # Load processor (handles image preprocessing)
            # Design Choice: Separate processor for clean separation of concerns
            # What This Enables: Standardized preprocessing (resize, normalize)
            # Why This Way: transformers best practice, handles all preprocessing
            processor_start = time.time()
            logger.info(f"⏱️  [BLIP] Loading processor from {model_name}...")
            self._processor = BlipProcessor.from_pretrained(model_name)  # type: ignore[assignment]
            logger.info(
                f"✓ [BLIP] Processor loaded in {time.time() - processor_start:.2f}s",
            )

            # Load model with appropriate dtype
            # Design Choice: float16 on GPU for speed, float32 on CPU for stability
            # What This Enables: 2x faster GPU inference, stable CPU inference
            # Why This Way: float16 cuts memory in half, safe for inference

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
                    "avoids CVE-2025-32434 and 76x faster loading)",
                )

            model_load_start = time.time()
            logger.info(
                "⏱️  [BLIP] Loading model weights (this may take 5-30s on CPU)...",
            )
            self._model = BlipForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
                use_safetensors=use_safetensors,
            )
            logger.info(
                f"✓ [BLIP] Model weights loaded in "
                f"{time.time() - model_load_start:.2f}s",
            )

            device_move_start = time.time()
            logger.info(f"⏱️  [BLIP] Moving model to {self._device}...")
            self._model.to(self._device)  # type: ignore[attr-defined]
            self._model.eval()  # type: ignore[attr-defined]  # Inference mode (disable dropout)
            logger.info(
                f"✓ [BLIP] Model ready on {self._device} in "
                f"{time.time() - device_move_start:.2f}s",
            )

            total_time = time.time() - load_start
            logger.info(
                f"✅ [BLIP] Total model load time: {total_time:.2f}s "
                f"(device={self._device})",
            )

        except ImportError as e:
            msg = (
                "transformers not installed. Install with: "
                "pip install transformers torch"
            )
            raise CaptionError(msg) from e
        except Exception as e:
            msg = "Failed to load BLIP model"
            raise CaptionError(
                msg,
                context={"device": self._device, "error": str(e)},
            ) from e

    def caption(self, image_bytes: bytes, max_length: int = 100) -> str:
        """Generate caption for image using BLIP.

        Design Choices:
        - **SHA256 cache key**: Content-addressed caching (same image = same caption)
        - **Check cache first**: Instant return if image seen before
        - **Beam search (num_beams=5)**: Better quality than greedy decoding
        - **torch.no_grad()**: Disable gradients for inference (faster, less memory)

        What This Enables:
        - Instant results for repeated images (0ms vs 500ms-1s)
        - High quality captions (beam search explores multiple hypotheses)
        - Efficient memory usage (no_grad saves ~30% memory)
        - Deterministic caching (same bytes always produce same key)

        Why This Way:
        - SHA256 is standard for content addressing (git uses this)
        - Cache-first prevents unnecessary computation
        - Beam search is standard for caption quality
        - no_grad() is PyTorch best practice for inference

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
        >>> captioner = BLIPCaptionTool()
        >>> caption = captioner.caption(image_bytes)
        >>> print(caption)
        'a person sitting at a desk with a laptop'

        >>> # Short captions
        >>> caption = captioner.caption(image_bytes, max_length=50)

        >>> # Cached result (instant)
        >>> caption2 = captioner.caption(image_bytes)  # Returns immediately
        """
        import time

        caption_start = time.time()

        # Check cache first
        # Design Choice: SHA256 hash of raw bytes as cache key
        # What This Enables: Same image = same key, works across restarts if persisted
        # Why This Way: Content-addressed caching is deterministic and reliable
        cache_key = hashlib.sha256(image_bytes).hexdigest()
        if cache_key in self._cache:
            logger.info("⚡ [BLIP] Caption cache hit (0ms)")
            return self._cache[cache_key]

        logger.info(
            f"⏱️  [BLIP] Starting caption generation "
            f"(image_size={len(image_bytes)} bytes)",
        )

        # Lazy load model if not loaded
        if self._model is None:
            logger.info("⏱️  [BLIP] Model not loaded yet, loading now...")
            self._load_model()

        try:
            # Convert bytes → PIL Image
            # Design Choice: PIL for compatibility with transformers
            # What This Enables: Handles all image formats (JPEG, PNG, etc.)
            # Why This Way: transformers expects PIL images
            image_load_start = time.time()
            image = Image.open(io.BytesIO(image_bytes))
            logger.info(
                f"✓ [BLIP] Image loaded in "
                f"{(time.time() - image_load_start) * 1000:.1f}ms",
            )

            # Preprocess image
            # Design Choice: Use processor for standardized preprocessing
            # What This Enables: Correct image format (resize, normalize, tensor)
            # Why This Way: Processor handles all model-specific requirements
            preprocess_start = time.time()
            logger.info("⏱️  [BLIP] Preprocessing image...")
            inputs = self._processor(image, return_tensors="pt").to(self._device)  # type: ignore[misc]
            logger.info(
                f"✓ [BLIP] Preprocessing done in "
                f"{(time.time() - preprocess_start) * 1000:.1f}ms",
            )

            # Generate caption with beam search
            # Design Choice: num_beams=5 for quality, max_length for control
            # What This Enables: Better captions than greedy (explores options)
            # Why This Way: Beam search is standard for seq2seq quality
            generation_start = time.time()
            logger.info(
                f"⏱️  [BLIP] Generating caption with beam search "
                f"(max_length={max_length})...",
            )
            with torch.no_grad():  # Disable gradients for inference
                output = self._model.generate(  # type: ignore[attr-defined]
                    **inputs,
                    max_length=max_length,
                    num_beams=2,  # Beam search for better quality
                )
            generation_time = time.time() - generation_start
            logger.info(f"✓ [BLIP] Caption generated in {generation_time:.2f}s")

            # Decode output tokens → text
            # Design Choice: skip_special_tokens=True removes [CLS], [SEP], etc.
            # What This Enables: Clean caption text, no internal tokens
            # Why This Way: Users don't need to see model internals
            decode_start = time.time()
            caption = self._processor.decode(output[0], skip_special_tokens=True)  # type: ignore[attr-defined]
            logger.info(
                f"✓ [BLIP] Decoded in {(time.time() - decode_start) * 1000:.1f}ms",
            )

            # Cache result
            # Design Choice: Store in memory for instant future lookups
            # What This Enables: 0ms retrieval for repeated images
            # Why This Way: Captions are tiny (~50-200 chars), memory is cheap
            self._cache[cache_key] = caption

            total_time = time.time() - caption_start
            preview = caption[:50]
            suffix = "..." if len(caption) > 50 else ""
            logger.info(
                f'✅ [BLIP] Caption complete in {total_time:.2f}s: "{preview}{suffix}"',
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
