import voyageai
from typing import List
from app.core.config import get_settings
import logging

logger = logging.getLogger(__name__)


class Embedder:
    """Generate embeddings using Voyage AI's embedding model."""

    def __init__(self):
        settings = get_settings()
        self.client = voyageai.AsyncClient(api_key=settings.voyage_api_key)
        self.model = settings.voyage_embedding_model

    async def embed_text(self, text: str, input_type: str = "query") -> List[float]:
        """Generate embedding for a single text."""
        try:
            result = await self.client.embed([text], model=self.model, input_type=input_type)
            return result.embeddings[0]
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return []

    async def embed_texts(self, texts: List[str], input_type: str = "document") -> List[List[float]]:
        """Generate embeddings for multiple texts (batch)."""
        try:
            result = await self.client.embed(texts, model=self.model, input_type=input_type)
            return result.embeddings
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            return []
