import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def app_client():
    """TestClient with all external dependencies mocked."""
    with (
        patch("app.api.routes.chat.Retriever") as mock_retriever_cls,
        patch("app.api.routes.chat.LLMClient") as mock_llm_cls,
        patch("app.api.routes.chat.RedisManager") as mock_redis_cls,
        patch("app.api.routes.chat.LangfuseManager") as mock_langfuse_cls,
        patch("app.api.routes.chat.InputGuard") as mock_guard_cls,
    ):
        # Retriever mock
        retriever_instance = MagicMock()
        retriever_instance.retrieve = AsyncMock(return_value=[
            {
                "id": 10,
                "score": 0.92,
                "payload": {
                    "digimon_id": 1,
                    "name": "Agumon",
                    "chunk_text": "Agumon is a Rookie level Digimon.",
                },
            }
        ])
        retriever_instance.close = AsyncMock()
        mock_retriever_cls.return_value = retriever_instance

        # LLMClient mock
        llm_instance = MagicMock()
        llm_instance.chat = AsyncMock(return_value={
            "content": "Agumon is a Rookie Digimon.",
            "model": "claude-haiku-4-5-20251001",
            "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
        })
        llm_instance.close = AsyncMock()
        mock_llm_cls.return_value = llm_instance

        # Guardrail mock — always allow through in tests
        guard_instance = MagicMock()
        guard_instance.is_digimon_related = AsyncMock(return_value=True)
        mock_guard_cls.return_value = guard_instance

        # Redis mock — cache miss by default
        redis_instance = MagicMock()
        redis_instance.get = AsyncMock(return_value=None)
        redis_instance.set = AsyncMock()
        redis_instance.close = AsyncMock()
        mock_redis_cls.return_value = redis_instance

        # Langfuse mock — trace methods are sync (no await)
        langfuse_instance = MagicMock()
        langfuse_instance.trace_retrieval = MagicMock()
        langfuse_instance.trace_chat = MagicMock()
        langfuse_instance.flush = MagicMock()
        mock_langfuse_cls.return_value = langfuse_instance

        from app.api.main import app
        with TestClient(app) as client:
            yield client, redis_instance


def test_chat_returns_200(app_client):
    client, _ = app_client
    response = client.post("/api/v1/chat", json={"query": "Tell me about Agumon"})
    assert response.status_code == 200


def test_chat_response_structure(app_client):
    client, _ = app_client
    response = client.post("/api/v1/chat", json={"query": "Tell me about Agumon"})
    data = response.json()
    assert "response" in data
    assert "context_sources" in data
    assert "token_usage" in data
    assert "latency_ms" in data


def test_chat_cache_hit(app_client):
    client, redis_instance = app_client
    cached = {
        "response": "Cached answer about Agumon.",
        "context_sources": [],
        "token_usage": {},
        "latency_ms": 5,
    }
    redis_instance.get = AsyncMock(return_value=cached)

    response = client.post("/api/v1/chat", json={"query": "cache me"})
    assert response.status_code == 200
    assert response.json()["response"] == "Cached answer about Agumon."

    # Reset to cache miss for other tests
    redis_instance.get = AsyncMock(return_value=None)


def test_chat_with_filters(app_client):
    client, _ = app_client
    response = client.post(
        "/api/v1/chat",
        json={"query": "Rookie Digimon list", "level_filter": "Rookie"},
    )
    assert response.status_code == 200
