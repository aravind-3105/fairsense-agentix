"""Caching service for reducing redundant expensive operations.

This module provides content-addressed caching to avoid redundant:
- LLM API calls
- OCR operations
- Image captioning
- Embedding computations

The cache supports multiple backends (memory, filesystem, redis) and provides
automatic cache key generation based on content hashing.

Example:
    >>> from fairsense_agentix.services.cache import cache
    >>> key = cache.hash_inputs(content="Hello world", model="gpt-4", version="1.0")
    >>> result = cache.get(key)
    >>> if result is None:
    ...     result = expensive_llm_call()
    ...     cache.put(key, result)
"""

import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any

from fairsense_agentix.configs import settings
from fairsense_agentix.services.telemetry import telemetry


class CacheBackend:
    """Base interface for cache backends.

    All cache backends must implement get, put, delete, and clear methods.
    """

    def get(self, key: str) -> Any | None:
        """Retrieve a value from cache.

        Parameters
        ----------
        key : str
            Cache key

        Returns
        -------
        Any | None
            Cached value or None if not found/expired
        """
        raise NotImplementedError

    def put(self, key: str, value: Any, ttl: int) -> None:
        """Store a value in cache.

        Parameters
        ----------
        key : str
            Cache key
        value : Any
            Value to cache (must be pickle-able)
        ttl : int
            Time-to-live in seconds
        """
        raise NotImplementedError

    def delete(self, key: str) -> None:
        """Delete a key from cache.

        Parameters
        ----------
        key : str
            Cache key to delete
        """
        raise NotImplementedError

    def clear(self) -> None:
        """Clear all entries from cache."""
        raise NotImplementedError


class MemoryCacheBackend(CacheBackend):
    """In-memory cache backend using a dictionary.

    Simple, fast, but not persistent across process restarts.
    Suitable for development and single-process deployments.
    """

    def __init__(self) -> None:
        """Initialize the memory cache."""
        self._cache: dict[str, tuple[Any, float]] = {}

    def get(self, key: str) -> Any | None:
        """Retrieve value from memory cache.

        Parameters
        ----------
        key : str
            Cache key

        Returns
        -------
        Any | None
            Cached value or None if not found/expired
        """
        if key not in self._cache:
            return None

        value, expiry = self._cache[key]

        # Check if expired
        if expiry < time.time():
            del self._cache[key]
            return None

        return value

    def put(self, key: str, value: Any, ttl: int) -> None:
        """Store value in memory cache.

        Parameters
        ----------
        key : str
            Cache key
        value : Any
            Value to cache
        ttl : int
            Time-to-live in seconds
        """
        expiry = time.time() + ttl
        self._cache[key] = (value, expiry)

    def delete(self, key: str) -> None:
        """Delete key from memory cache.

        Parameters
        ----------
        key : str
            Cache key to delete
        """
        self._cache.pop(key, None)

    def clear(self) -> None:
        """Clear all entries from memory cache."""
        self._cache.clear()


class FilesystemCacheBackend(CacheBackend):
    """Filesystem-based cache backend.

    Stores cached values as pickle files on disk.
    Persistent across process restarts, suitable for local development.
    """

    def __init__(self, cache_dir: Path) -> None:
        """Initialize the filesystem cache.

        Parameters
        ----------
        cache_dir : Path
            Directory to store cache files
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, key: str) -> Path:
        """Get the file path for a cache key.

        Parameters
        ----------
        key : str
            Cache key

        Returns
        -------
        Path
            Path to cache file
        """
        return self.cache_dir / f"{key}.cache"

    def get(self, key: str) -> Any | None:
        """Retrieve value from filesystem cache.

        Parameters
        ----------
        key : str
            Cache key

        Returns
        -------
        Any | None
            Cached value or None if not found/expired
        """
        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)

            value, expiry = data["value"], data["expiry"]

            # Check if expired
            if expiry < time.time():
                cache_path.unlink(missing_ok=True)
                return None

            return value

        except (pickle.PickleError, KeyError, OSError):
            # Corrupted cache file, delete it
            cache_path.unlink(missing_ok=True)
            return None

    def put(self, key: str, value: Any, ttl: int) -> None:
        """Store value in filesystem cache.

        Parameters
        ----------
        key : str
            Cache key
        value : Any
            Value to cache (must be pickle-able)
        ttl : int
            Time-to-live in seconds
        """
        cache_path = self._get_cache_path(key)
        expiry = time.time() + ttl

        data = {"value": value, "expiry": expiry}

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
        except (pickle.PickleError, OSError) as e:
            telemetry.log_warning("cache_write_failed", key=key, error=str(e))

    def delete(self, key: str) -> None:
        """Delete key from filesystem cache.

        Parameters
        ----------
        key : str
            Cache key to delete
        """
        cache_path = self._get_cache_path(key)
        cache_path.unlink(missing_ok=True)

    def clear(self) -> None:
        """Clear all entries from filesystem cache."""
        for cache_file in self.cache_dir.glob("*.cache"):
            cache_file.unlink(missing_ok=True)


class CacheService:
    """High-level caching service with automatic key generation.

    This service provides a simple interface for caching expensive operations.
    It automatically generates cache keys based on content hashing and manages
    TTL and invalidation.

    Attributes
    ----------
    enabled : bool
        Whether caching is enabled globally
    backend : CacheBackend
        The cache backend implementation
    default_ttl : int
        Default time-to-live for cache entries

    """

    def __init__(
        self,
        enabled: bool = True,
        backend: str = "memory",
        cache_dir: Path = Path(".cache"),
        ttl: int = 3600,
    ) -> None:
        """Initialize the cache service.

        Parameters
        ----------
        enabled : bool, optional
            Whether to enable caching, by default True
        backend : str, optional
            Backend type ("memory", "filesystem", "redis"), by default "memory"
        cache_dir : Path, optional
            Directory for filesystem cache, by default Path(".cache")
        ttl : int, optional
            Default TTL in seconds, by default 3600
        """
        self.enabled = enabled
        self.default_ttl = ttl

        # Initialize backend
        if backend == "memory":
            self.backend: CacheBackend = MemoryCacheBackend()
        elif backend == "filesystem":
            self.backend = FilesystemCacheBackend(cache_dir)
        elif backend == "redis":
            # Placeholder for future Redis implementation
            msg = "Redis backend not yet implemented, falling back to memory"
            telemetry.log_warning("cache_backend_fallback", backend=backend)
            self.backend = MemoryCacheBackend()
        else:
            msg = f"Unknown cache backend: {backend}"
            raise ValueError(msg)

    def hash_inputs(self, **kwargs: Any) -> str:
        """Generate a content-addressed cache key from inputs.

        This creates a deterministic hash based on all provided arguments.
        The hash changes if ANY input changes, ensuring cache invalidation
        when models, prompts, or content change.

        Parameters
        ----------
        **kwargs : Any
            All inputs that affect the output (content, model, version, etc.)

        Returns
        -------
        str
            SHA256 hash of the inputs

        Example
        -------
        >>> key = cache.hash_inputs(
        ...     content=image_bytes, model="tesseract", version="5.0", language="eng"
        ... )
        """
        # Sort keys for deterministic hashing
        sorted_items = sorted(kwargs.items())

        # Convert to JSON (handles most types)
        # For bytes, convert to hex string
        serializable = []
        for k, v in sorted_items:
            if isinstance(v, bytes):
                serializable.append((k, v.hex()))
            else:
                serializable.append((k, v))

        content = json.dumps(serializable, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, key: str, operation: str = "unknown") -> Any | None:
        """Retrieve a value from cache.

        Parameters
        ----------
        key : str
            Cache key (from hash_inputs)
        operation : str, optional
            Operation name for telemetry, by default "unknown"

        Returns
        -------
        Any | None
            Cached value or None if not found/disabled
        """
        if not self.enabled:
            return None

        result = self.backend.get(key)

        if result is not None:
            telemetry.record_cache_hit(operation, key)
        else:
            telemetry.record_cache_miss(operation, key)

        return result

    def put(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
        operation: str = "unknown",
    ) -> None:
        """Store a value in cache.

        Parameters
        ----------
        key : str
            Cache key (from hash_inputs)
        value : Any
            Value to cache (must be pickle-able)
        ttl : int | None, optional
            Time-to-live in seconds, uses default if None
        operation : str, optional
            Operation name for telemetry, by default "unknown"
        """
        if not self.enabled:
            return

        ttl = ttl if ttl is not None else self.default_ttl
        self.backend.put(key, value, ttl)

        telemetry.log_info(
            "cache_write",
            operation=operation,
            key=key[:16] + "...",
            ttl=ttl,
        )

    def delete(self, key: str) -> None:
        """Delete a key from cache.

        Parameters
        ----------
        key : str
            Cache key to delete
        """
        if not self.enabled:
            return

        self.backend.delete(key)

    def clear(self) -> None:
        """Clear all entries from cache."""
        if not self.enabled:
            return

        self.backend.clear()
        telemetry.log_info("cache_cleared")


# Singleton instance configured from settings
cache = CacheService(
    enabled=settings.cache_enabled,
    backend=settings.cache_backend,
    cache_dir=settings.cache_dir,
    ttl=settings.cache_ttl_seconds,
)
