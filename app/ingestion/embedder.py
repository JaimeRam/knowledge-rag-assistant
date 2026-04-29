from openai import AsyncOpenAI
from typing import List
from app.core.config import get_settings
import logging

logger = logging.getLogger(__name__)


class Embedder:
    """Generate embeddings using OpenAI's embedding model."""
    
    def __init__(self):
        settings = get_settings()
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_embedding_model
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return []
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts (batch)."""
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            return []
    
    async def close(self):
        """Close the OpenAI client."""
        await self.client.close()
