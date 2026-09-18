"""Тесты конфигурации: значения по умолчанию, переменные окружения, инварианты."""
import importlib
import os

import pytest


@pytest.fixture
def reload_config(monkeypatch):
    """Перезагружает config.py после изменения переменных окружения в тесте."""
    def _reload():
        import config
        return importlib.reload(config)
    return _reload


def test_default_values_when_env_not_set(monkeypatch, reload_config):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("QDRANT_URL", raising=False)

    config = reload_config()

    assert config.TELEGRAM_BOT_TOKEN == "YOUR_BOT_TOKEN_HERE"
    assert config.OLLAMA_BASE_URL == "http://localhost:11434"
    assert config.QDRANT_URL == ":memory:"


def test_env_vars_override_defaults(monkeypatch, reload_config):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token-123")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.internal:11434")
    monkeypatch.setenv("QDRANT_URL", "http://qdrant.internal:6333")

    config = reload_config()

    assert config.TELEGRAM_BOT_TOKEN == "test-token-123"
    assert config.OLLAMA_BASE_URL == "http://ollama.internal:11434"
    assert config.QDRANT_URL == "http://qdrant.internal:6333"


def test_chunk_overlap_smaller_than_chunk_size(reload_config):
    # Инвариант langchain-сплиттера: overlap не может быть >= chunk_size,
    # иначе RecursiveCharacterTextSplitter упадёт с ValueError при разбиении.
    config = reload_config()
    assert config.CHUNK_OVERLAP < config.CHUNK_SIZE


def test_max_history_length_is_positive(reload_config):
    config = reload_config()
    assert config.MAX_HISTORY_LENGTH > 0
