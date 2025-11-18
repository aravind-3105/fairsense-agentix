"""Unit tests for Settings configuration."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from fairsense_agentix.configs.settings import Settings, settings


pytestmark = pytest.mark.unit


class TestSettings:
    """Test suite for Settings class."""

    def test_default_settings(self):
        """Test that Settings can be instantiated with defaults."""
        settings = Settings()

        assert settings.llm_provider == "fake"
        assert settings.llm_model_name == "gpt-4"
        assert settings.llm_temperature == 0.3
        assert settings.cache_enabled is True
        assert settings.telemetry_enabled is False
        assert settings.api_port == 8000

    def test_env_override_simple(self, monkeypatch):
        """Test that environment variables override defaults."""
        monkeypatch.setenv("FAIRSENSE_LLM_PROVIDER", "openai")
        monkeypatch.setenv("FAIRSENSE_LLM_MODEL_NAME", "gpt-4-turbo")
        monkeypatch.setenv("FAIRSENSE_API_PORT", "9000")

        # Note: Need to provide API key for openai provider
        monkeypatch.setenv("FAIRSENSE_LLM_API_KEY", "sk-test-key")

        settings = Settings()

        assert settings.llm_provider == "openai"
        assert settings.llm_model_name == "gpt-4-turbo"
        assert settings.api_port == 9000
        assert settings.llm_api_key == "sk-test-key"

    def test_env_override_boolean(self, monkeypatch):
        """Test that boolean env vars are parsed correctly."""
        monkeypatch.setenv("FAIRSENSE_CACHE_ENABLED", "false")
        monkeypatch.setenv("FAIRSENSE_TELEMETRY_ENABLED", "true")

        settings = Settings()

        assert settings.cache_enabled is False
        assert settings.telemetry_enabled is True

    def test_env_override_nested_types(self, monkeypatch):
        """Test that complex types like floats are parsed correctly."""
        monkeypatch.setenv("FAIRSENSE_LLM_TEMPERATURE", "0.7")
        monkeypatch.setenv("FAIRSENSE_ROUTER_CONFIDENCE_THRESHOLD", "0.85")

        settings = Settings()

        assert settings.llm_temperature == 0.7
        assert settings.router_confidence_threshold == 0.85

    def test_invalid_llm_temperature_range(self):
        """Test that temperature out of range raises ValidationError."""
        with pytest.raises(ValidationError, match="less than or equal to 1"):
            Settings(llm_temperature=1.5)

        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            Settings(llm_temperature=-0.1)

    def test_invalid_api_port_range(self):
        """Test that invalid port raises ValidationError."""
        with pytest.raises(ValidationError, match="less than or equal to 65535"):
            Settings(api_port=70000)

        with pytest.raises(ValidationError, match="greater than 0"):
            Settings(api_port=0)

    def test_api_key_required_for_openai(self):
        """Test that API key is required when using openai provider."""
        with pytest.raises(
            ValidationError,
            match="llm_api_key is required when llm_provider=openai",
        ):
            Settings(llm_provider="openai", llm_api_key=None)

    def test_api_key_required_for_anthropic(self):
        """Test that API key is required when using anthropic provider."""
        with pytest.raises(
            ValidationError,
            match="llm_api_key is required when llm_provider=anthropic",
        ):
            Settings(llm_provider="anthropic", llm_api_key=None)

    def test_api_key_not_required_for_fake(self):
        """Test that API key is not required for fake provider."""
        settings = Settings(llm_provider="fake", llm_api_key=None)
        assert settings.llm_api_key is None

    def test_path_expansion(self):
        """Test that paths are expanded to absolute paths."""
        settings = Settings(
            faiss_risks_index_path=Path("data/indexes/risks.faiss"),
            cache_dir=Path(".cache"),
        )

        # Paths should be absolute
        assert settings.faiss_risks_index_path.is_absolute()
        assert settings.cache_dir.is_absolute()

    def test_literal_validation(self):
        """Test that Literal fields reject invalid values."""
        with pytest.raises(ValidationError, match="Input should be"):
            Settings(llm_provider="invalid_provider")

        with pytest.raises(ValidationError, match="Input should be"):
            Settings(cache_backend="invalid_backend")

        with pytest.raises(ValidationError, match="Input should be"):
            Settings(log_level="INVALID")

    def test_case_insensitive_env_vars(self, monkeypatch):
        """Test that env vars are case insensitive."""
        # Pydantic settings are case insensitive by default (case_sensitive=False)
        monkeypatch.setenv("fairsense_api_port", "7000")

        settings = Settings()
        assert settings.api_port == 7000

    def test_extra_env_vars_ignored(self, monkeypatch):
        """Test that extra env vars don't cause errors."""
        monkeypatch.setenv("FAIRSENSE_NONEXISTENT_FIELD", "value")

        # Should not raise an error due to extra="ignore"
        settings = Settings()
        assert not hasattr(settings, "nonexistent_field")

    def test_workflow_timeout_positive(self):
        """Test that workflow timeout must be positive."""
        with pytest.raises(ValidationError, match="greater than 0"):
            Settings(workflow_timeout_seconds=0)

        with pytest.raises(ValidationError, match="greater than 0"):
            Settings(workflow_timeout_seconds=-100)

    def test_max_refinement_iterations_non_negative(self):
        """Test that max refinement iterations can be zero or positive."""
        # Should accept 0 (no refinement)
        settings = Settings(max_refinement_iterations=0)
        assert settings.max_refinement_iterations == 0

        # Should accept positive values
        settings = Settings(max_refinement_iterations=5)
        assert settings.max_refinement_iterations == 5

        # Should reject negative
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            Settings(max_refinement_iterations=-1)

    def test_settings_singleton(self):
        """Test that the singleton instance works."""
        assert isinstance(settings, Settings)
        assert settings.llm_provider in ["openai", "anthropic", "local", "fake"]
