from app.db.qdrant import QdrantManager
from app.ingestion.embedder import Embedder
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Retriever:
    """Retrieve relevant Digimon information from Qdrant."""
    
    def __init__(self):
        self.qdrant = QdrantManager()
        self.embedder = Embedder()
    
    async def retrieve(self, query: str, limit: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks for a query."""
        # Generate embedding for query
        query_vector = await self.embedder.embed_text(query, input_type="query")
        
        if not query_vector:
            logger.error("Failed to generate query embedding")
            return []
        
        # Search in Qdrant
        results = await self.qdrant.search(
            query_vector=query_vector,
            limit=limit,
            score_threshold=0.5,
            filter_params=filters
        )
        
        logger.info(f"Retrieved {len(results)} chunks for query: {query}")
        return results
    
    async def close(self):
        """Close resources."""
        pass
