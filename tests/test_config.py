"""
Tests for core.config module.
Tests: load_config, environment variable overrides, default values.
"""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from core.config import load_config, CaseCraftConfig


class TestLoadConfig:
    """Tests for load_config function."""

    def test_returns_default_config_without_file(self):
        """When no config file exists, defaults should be used."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Point to a non-existent config file
            config = load_config(os.path.join(tmpdir, "nonexistent.yaml"))
            
            assert config.general.model == "llama3.1:8b"
            assert config.general.llm_provider == "ollama"
            assert config.general.timeout == 600
            assert config.generation.chunk_size == 1000
            assert config.quality.similarity_threshold == 0.85

    def test_loads_from_yaml_file(self):
        """Config should be loaded from YAML file."""
        yaml_content = """
general:
  model: custom-model
  timeout: 300
generation:
  chunk_size: 2000
"""
        # Use delete_on_close=False for Windows compatibility
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        try:
            f.write(yaml_content)
            f.close()  # Close before reading on Windows
            
            config = load_config(f.name)
            
            assert config.general.model == "custom-model"
            assert config.general.timeout == 300
            assert config.generation.chunk_size == 2000
            # Non-specified values should use defaults
            assert config.generation.temperature == 0.2
        finally:
            try:
                os.unlink(f.name)
            except OSError:
                pass  # Ignore cleanup errors

    def test_env_vars_override_yaml(self):
        """Environment variables should override YAML config."""
        yaml_content = """
general:
  model: yaml-model
  timeout: 300
"""
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        try:
            f.write(yaml_content)
            f.close()  # Close before reading on Windows
            
            # Set environment variable (use correct naming: CASECRAFT_GENERAL_MODEL)
            with patch.dict(os.environ, {'CASECRAFT_GENERAL_MODEL': 'env-model'}):
                config = load_config(f.name)
                
                # Env var should win
                assert config.general.model == "env-model"
                # YAML value should still apply for non-overridden keys
                assert config.general.timeout == 300
        finally:
            try:
                os.unlink(f.name)
            except OSError:
                pass  # Ignore cleanup errors

    def test_default_values_applied(self):
        """All default values should be applied correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = load_config(os.path.join(tmpdir, "nonexistent.yaml"))
            
            # General settings
            assert config.general.base_url == "http://localhost:11434"
            assert config.general.max_retries == 2
            assert config.general.context_window_ratio == 0.75
            assert config.general.output_token_ratio == 0.25
            assert config.general.min_output_tokens == 1024
            
            # Generation settings
            assert config.generation.chunk_overlap == 100
            assert config.generation.temperature == 0.2
            assert config.generation.top_p == 0.9
            assert config.generation.app_type == "web"
            assert config.generation.min_cases_per_chunk == 5
            
            # Quality settings
            assert config.quality.semantic_deduplication is True
            assert config.quality.reviewer_pass is True
            assert config.quality.top_k == 5
            
            # Knowledge settings
            assert config.knowledge.kb_chunk_size == 1500
            assert config.knowledge.query_decomposition is True
            assert config.knowledge.query_expansion is True


class TestConfigModel:
    """Tests for CaseCraftConfig model validation."""

    def test_config_model_has_expected_sections(self):
        config = CaseCraftConfig()
        
        assert hasattr(config, 'general')
        assert hasattr(config, 'generation')
        assert hasattr(config, 'output')
        assert hasattr(config, 'quality')
        assert hasattr(config, 'knowledge')

    def test_secret_str_for_api_key(self):
        """API key should be stored as SecretStr."""
        config = CaseCraftConfig()
        
        # SecretStr should not reveal value directly
        api_key_str = str(config.general.api_key)
        assert "**********" in api_key_str or "ollama" not in api_key_str
        
        # But we should be able to get the actual value
        assert config.general.api_key.get_secret_value() == "ollama"
