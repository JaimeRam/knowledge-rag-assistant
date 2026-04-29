from openai import AsyncOpenAI
from typing import Dict, Any
from app.core.config import get_settings
import logging

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for OpenAI LLM API."""
    
    def __init__(self):
        settings = get_settings()
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
    
    async def chat(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Send a chat completion request."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            
            return {
                "content": response.choices[0].message.content,
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
        except Exception as e:
            logger.error(f"Error in LLM chat: {e}")
            return {"content": "Sorry, I encountered an error processing your request.", "usage": {}}
    
    async def close(self):
        """Close the OpenAI client."""
        await self.client.close()
