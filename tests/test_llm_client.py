import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def mock_anthropic_client():
    """Patch AsyncAnthropic at the import location in llm_client."""
    with patch("app.rag.llm_client.anthropic.AsyncAnthropic") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        yield mock_instance


async def test_chat_returns_content(mock_anthropic_client):
    from app.rag.llm_client import LLMClient

    fake_message = MagicMock()
    fake_message.content = [MagicMock(text="Agumon is a Rookie Digimon.")]
    fake_message.model = "claude-haiku-4-5-20251001"
    fake_message.usage = MagicMock(input_tokens=30, output_tokens=10)
    mock_anthropic_client.messages.create = AsyncMock(return_value=fake_message)

    client = LLMClient()
    result = await client.chat("What is Agumon?")

    assert result["content"] == "Agumon is a Rookie Digimon."
    assert "usage" in result
    assert result["usage"]["total_tokens"] == 40


async def test_chat_returns_token_usage(mock_anthropic_client):
    from app.rag.llm_client import LLMClient

    fake_message = MagicMock()
    fake_message.content = [MagicMock(text="OK")]
    fake_message.model = "claude-haiku-4-5-20251001"
    fake_message.usage = MagicMock(input_tokens=10, output_tokens=5)
    mock_anthropic_client.messages.create = AsyncMock(return_value=fake_message)

    client = LLMClient()
    result = await client.chat("test")

    assert result["usage"]["prompt_tokens"] == 10
    assert result["usage"]["completion_tokens"] == 5


async def test_chat_handles_api_error(mock_anthropic_client):
    from app.rag.llm_client import LLMClient

    mock_anthropic_client.messages.create = AsyncMock(side_effect=Exception("API error"))

    client = LLMClient()
    with pytest.raises(Exception, match="API error"):
        await client.chat("any question")
