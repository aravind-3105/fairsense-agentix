"""Unit tests for Cache service."""

import time

import pytest

from fairsense_agentix.services.cache import (
    CacheService,
    FilesystemCacheBackend,
    MemoryCacheBackend,
)


class TestMemoryCacheBackend:
    """Test suite for MemoryCacheBackend."""

    def test_init(self):
        """Test that memory cache can be initialized."""
        backend = MemoryCacheBackend()
        assert backend._cache == {}

    def test_put_and_get(self):
        """Test basic put and get operations."""
        backend = MemoryCacheBackend()

        backend.put("key1", "value1", ttl=60)
        result = backend.get("key1")

        assert result == "value1"

    def test_get_nonexistent_key(self):
        """Test that getting nonexistent key returns None."""
        backend = MemoryCacheBackend()

        result = backend.get("nonexistent")
        assert result is None

    def test_get_expired_entry(self):
        """Test that expired entries return None."""
        backend = MemoryCacheBackend()

        backend.put("key1", "value1", ttl=1)  # 1 second TTL
        time.sleep(1.1)  # Wait for expiry

        result = backend.get("key1")
        assert result is None

    def test_delete(self):
        """Test deleting a key."""
        backend = MemoryCacheBackend()

        backend.put("key1", "value1", ttl=60)
        backend.delete("key1")

        result = backend.get("key1")
        assert result is None

    def test_delete_nonexistent_key(self):
        """Test that deleting nonexistent key doesn't raise error."""
        backend = MemoryCacheBackend()

        # Should not raise an error
        backend.delete("nonexistent")

    def test_clear(self):
        """Test clearing all entries."""
        backend = MemoryCacheBackend()

        backend.put("key1", "value1", ttl=60)
        backend.put("key2", "value2", ttl=60)
        backend.put("key3", "value3", ttl=60)

        backend.clear()

        assert backend.get("key1") is None
        assert backend.get("key2") is None
        assert backend.get("key3") is None

    def test_complex_values(self):
        """Test that complex Python objects can be cached."""
        backend = MemoryCacheBackend()

        complex_value = {
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "tuple": (4, 5, 6),
        }

        backend.put("complex", complex_value, ttl=60)
        result = backend.get("complex")

        assert result == complex_value


class TestFilesystemCacheBackend:
    """Test suite for FilesystemCacheBackend."""

    @pytest.fixture
    def temp_cache_dir(self, tmp_path):
        """Create a temporary cache directory for testing."""
        cache_dir = tmp_path / "test_cache"
        cache_dir.mkdir()
        return cache_dir

    def test_init_creates_directory(self, tmp_path):
        """Test that initialization creates cache directory."""
        cache_dir = tmp_path / "new_cache"
        FilesystemCacheBackend(cache_dir)

        assert cache_dir.exists()
        assert cache_dir.is_dir()

    def test_put_and_get(self, temp_cache_dir):
        """Test basic put and get operations."""
        backend = FilesystemCacheBackend(temp_cache_dir)

        backend.put("key1", "value1", ttl=60)
        result = backend.get("key1")

        assert result == "value1"

    def test_get_nonexistent_key(self, temp_cache_dir):
        """Test that getting nonexistent key returns None."""
        backend = FilesystemCacheBackend(temp_cache_dir)

        result = backend.get("nonexistent")
        assert result is None

    def test_get_expired_entry(self, temp_cache_dir):
        """Test that expired entries return None and file is deleted."""
        backend = FilesystemCacheBackend(temp_cache_dir)

        backend.put("key1", "value1", ttl=1)  # 1 second TTL
        cache_file = backend._get_cache_path("key1")
        assert cache_file.exists()

        time.sleep(1.1)  # Wait for expiry

        result = backend.get("key1")
        assert result is None
        assert not cache_file.exists()  # File should be deleted

    def test_delete(self, temp_cache_dir):
        """Test deleting a key."""
        backend = FilesystemCacheBackend(temp_cache_dir)

        backend.put("key1", "value1", ttl=60)
        cache_file = backend._get_cache_path("key1")
        assert cache_file.exists()

        backend.delete("key1")
        assert not cache_file.exists()

    def test_delete_nonexistent_key(self, temp_cache_dir):
        """Test that deleting nonexistent key doesn't raise error."""
        backend = FilesystemCacheBackend(temp_cache_dir)

        # Should not raise an error
        backend.delete("nonexistent")

    def test_clear(self, temp_cache_dir):
        """Test clearing all entries."""
        backend = FilesystemCacheBackend(temp_cache_dir)

        backend.put("key1", "value1", ttl=60)
        backend.put("key2", "value2", ttl=60)
        backend.put("key3", "value3", ttl=60)

        # Verify files exist
        assert len(list(temp_cache_dir.glob("*.cache"))) == 3

        backend.clear()

        # All cache files should be deleted
        assert len(list(temp_cache_dir.glob("*.cache"))) == 0

    def test_persistence_across_instances(self, temp_cache_dir):
        """Test that cache persists across backend instances."""
        backend1 = FilesystemCacheBackend(temp_cache_dir)
        backend1.put("persistent_key", "persistent_value", ttl=60)

        # Create new instance pointing to same directory
        backend2 = FilesystemCacheBackend(temp_cache_dir)
        result = backend2.get("persistent_key")

        assert result == "persistent_value"

    def test_complex_values(self, temp_cache_dir):
        """Test that complex Python objects can be cached."""
        backend = FilesystemCacheBackend(temp_cache_dir)

        complex_value = {
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "set": {4, 5, 6},  # Sets are pickle-able
        }

        backend.put("complex", complex_value, ttl=60)
        result = backend.get("complex")

        assert result["list"] == complex_value["list"]
        assert result["dict"] == complex_value["dict"]
        assert result["set"] == complex_value["set"]


class TestCacheService:
    """Test suite for CacheService."""

    def test_init_memory_backend(self):
        """Test initialization with memory backend."""
        service = CacheService(enabled=True, backend="memory")
        assert isinstance(service.backend, MemoryCacheBackend)

    def test_init_filesystem_backend(self, tmp_path):
        """Test initialization with filesystem backend."""
        cache_dir = tmp_path / "cache"
        service = CacheService(enabled=True, backend="filesystem", cache_dir=cache_dir)
        assert isinstance(service.backend, FilesystemCacheBackend)
        assert cache_dir.exists()

    def test_init_invalid_backend(self):
        """Test that invalid backend raises error."""
        with pytest.raises(ValueError, match="Unknown cache backend"):
            CacheService(enabled=True, backend="invalid")

    def test_hash_inputs_deterministic(self):
        """Test that hash_inputs produces deterministic keys."""
        service = CacheService(enabled=True, backend="memory")

        key1 = service.hash_inputs(content="test", model="gpt-4", temp=0.3)
        key2 = service.hash_inputs(content="test", model="gpt-4", temp=0.3)

        assert key1 == key2

    def test_hash_inputs_different_for_different_inputs(self):
        """Test that different inputs produce different keys."""
        service = CacheService(enabled=True, backend="memory")

        key1 = service.hash_inputs(content="test1", model="gpt-4")
        key2 = service.hash_inputs(content="test2", model="gpt-4")
        key3 = service.hash_inputs(content="test1", model="gpt-3")

        assert key1 != key2
        assert key1 != key3
        assert key2 != key3

    def test_hash_inputs_order_independent(self):
        """Test that argument order doesn't affect hash."""
        service = CacheService(enabled=True, backend="memory")

        key1 = service.hash_inputs(content="test", model="gpt-4", temp=0.3)
        key2 = service.hash_inputs(temp=0.3, model="gpt-4", content="test")

        assert key1 == key2

    def test_hash_inputs_with_bytes(self):
        """Test that bytes values can be hashed."""
        service = CacheService(enabled=True, backend="memory")

        key = service.hash_inputs(content=b"binary data", model="ocr")
        assert isinstance(key, str)
        assert len(key) == 64  # SHA256 produces 64 hex chars

    def test_get_put_roundtrip(self):
        """Test get/put roundtrip."""
        service = CacheService(enabled=True, backend="memory")

        key = service.hash_inputs(content="test")
        service.put(key, {"result": "data"})
        result = service.get(key)

        assert result == {"result": "data"}

    def test_get_when_disabled(self):
        """Test that get returns None when cache is disabled."""
        service = CacheService(enabled=False, backend="memory")

        key = service.hash_inputs(content="test")
        # Even though we put data, get should return None
        service.backend.put(key, "value", ttl=60)

        result = service.get(key)
        assert result is None

    def test_put_when_disabled(self):
        """Test that put is no-op when cache is disabled."""
        service = CacheService(enabled=False, backend="memory")

        key = service.hash_inputs(content="test")
        service.put(key, "value")

        # Should not actually be in backend
        result = service.backend.get(key)
        assert result is None

    def test_put_with_custom_ttl(self):
        """Test that custom TTL overrides default."""
        service = CacheService(enabled=True, backend="memory", ttl=60)

        key = service.hash_inputs(content="test")
        service.put(key, "value", ttl=1)  # 1 second TTL

        time.sleep(1.1)
        result = service.get(key)
        assert result is None

    def test_delete(self):
        """Test deleting a key."""
        service = CacheService(enabled=True, backend="memory")

        key = service.hash_inputs(content="test")
        service.put(key, "value")
        service.delete(key)

        result = service.get(key)
        assert result is None

    def test_delete_when_disabled(self):
        """Test that delete is no-op when disabled."""
        service = CacheService(enabled=False, backend="memory")

        # Should not raise an error
        key = service.hash_inputs(content="test")
        service.delete(key)

    def test_clear(self):
        """Test clearing all cache entries."""
        service = CacheService(enabled=True, backend="memory")

        key1 = service.hash_inputs(content="test1")
        key2 = service.hash_inputs(content="test2")

        service.put(key1, "value1")
        service.put(key2, "value2")

        service.clear()

        assert service.get(key1) is None
        assert service.get(key2) is None

    def test_clear_when_disabled(self):
        """Test that clear is no-op when disabled."""
        service = CacheService(enabled=False, backend="memory")

        # Should not raise an error
        service.clear()

    def test_operation_parameter_for_telemetry(self):
        """Test that operation parameter is accepted (for telemetry)."""
        service = CacheService(enabled=True, backend="memory")

        key = service.hash_inputs(content="test")
        service.put(key, "value", operation="test_op")
        result = service.get(key, operation="test_op")

        assert result == "value"
