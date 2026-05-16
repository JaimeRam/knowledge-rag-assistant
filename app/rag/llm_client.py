import anthropic
from typing import Dict, Any
from app.core.config import get_settings
import logging

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for Anthropic Claude API."""

    def __init__(self):
        settings = get_settings()
        self.client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        self.model = settings.anthropic_model

    async def chat(self, prompt: str, max_tokens: int = 1024, **kwargs) -> Dict[str, Any]:
        """Send a message and get a completion."""
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )

            return {
                "content": response.content[0].text,
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                },
            }
        except Exception as e:
            logger.error(f"Error in LLM chat: {e}")
            return {"content": "Sorry, I encountered an error processing your request.", "usage": {}}

    async def close(self):
        pass
