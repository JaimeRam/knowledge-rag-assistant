import pytest
from unittest.mock import AsyncMock, MagicMock


def _make_llm(response_text: str) -> MagicMock:
    llm = MagicMock()
    llm.chat = AsyncMock(return_value={"content": response_text, "usage": {}})
    return llm


async def test_digimon_query_passes():
    from app.guardrails.input_guard import InputGuard

    llm = _make_llm("YES")
    guard = InputGuard(llm)
    assert await guard.is_digimon_related("What are Agumon's skills?") is True


async def test_off_topic_query_blocked():
    from app.guardrails.input_guard import InputGuard

    llm = _make_llm("NO")
    guard = InputGuard(llm)
    assert await guard.is_digimon_related("What is the capital of France?") is False


async def test_fail_open_on_llm_error():
    from app.guardrails.input_guard import InputGuard

    llm = MagicMock()
    llm.chat = AsyncMock(side_effect=Exception("LLM unavailable"))
    guard = InputGuard(llm)
    # Must return True (allow through) to avoid cascading failure
    assert await guard.is_digimon_related("anything") is True


async def test_case_insensitive_yes():
    from app.guardrails.input_guard import InputGuard

    llm = _make_llm("  Yes  ")
    guard = InputGuard(llm)
    assert await guard.is_digimon_related("Tell me about Gabumon") is True
