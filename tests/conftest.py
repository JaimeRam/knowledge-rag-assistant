import pytest
from unittest.mock import patch, AsyncMock, MagicMock


@pytest.fixture(autouse=True, scope="session")
def mock_settings():
    """Mock settings for all tests — prevents loading .env or calling external services.

    Session-scoped and autouse because pydantic_agent.py calls get_settings() at import time.
    """
    from app.core.config import Settings

    fake = Settings(
        anthropic_api_key="test-anthropic-key",
        anthropic_model="claude-haiku-4-5-20251001",
        voyage_api_key="test-voyage-key",
        langfuse_public_key="",
        langfuse_secret_key="",
        langfuse_host="http://localhost:3000",
    )
    with patch("app.core.config.get_settings", return_value=fake):
        yield fake


@pytest.fixture
def sample_chunks():
    """Realistic Qdrant search result chunks."""
    return [
        {
            "id": 10,
            "score": 0.92,
            "payload": {
                "digimon_id": 1,
                "name": "Agumon",
                "chunk_index": 0,
                "chunk_text": "Agumon is a Rookie level Digimon with Reptile type and Vaccine attribute.",
                "level": "Rookie",
                "type": "Reptile",
                "attribute": "Vaccine",
            },
        },
        {
            "id": 11,
            "score": 0.85,
            "payload": {
                "digimon_id": 1,
                "name": "Agumon",
                "chunk_index": 1,
                "chunk_text": "Agumon: A Reptile Digimon that is well balanced in both offense and defense.",
                "level": "Rookie",
                "type": "Reptile",
                "attribute": "Vaccine",
            },
        },
    ]


@pytest.fixture
def mock_llm_response():
    """Minimal LLMClient.chat() return value."""
    return {
        "content": "Agumon is a Rookie level Digimon.",
        "model": "claude-haiku-4-5-20251001",
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 20,
            "total_tokens": 70,
        },
    }
