from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Any, Optional
from app.core.config import get_settings
import logging

logger = logging.getLogger(__name__)


class QdrantManager:
    def __init__(self):
        settings = get_settings()
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port
        )
        self.collection_name = settings.qdrant_collection_name
        self.vector_size = 1536  # OpenAI text-embedding-3-small

    async def create_collection(self) -> None:
        """Create collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created collection: {self.collection_name}")
        else:
            logger.info(f"Collection already exists: {self.collection_name}")

    async def upsert_points(
        self,
        points: List[PointStruct]
    ) -> None:
        """Upsert points to the collection."""
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        logger.info(f"Upserted {len(points)} points")

    async def search(
        self,
        query_vector: List[float],
        limit: int = 5,
        score_threshold: float = 0.5,
        filter_params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar points."""
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=filter_params
        )
        
        results = []
        for hit in search_result:
            results.append({
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload
            })
        
        return results

    async def delete_collection(self) -> None:
        """Delete the collection."""
        self.client.delete_collection(self.collection_name)
        logger.info(f"Deleted collection: {self.collection_name}")

    async def get_collection_info(self) -> Dict[str, Any]:
        """Get collection info."""
        info = self.client.get_collection(self.collection_name)
        return {
            "name": info.config.params.vectors.size,
            "points_count": info.points_count,
            "vector_size": info.config.params.vectors.size
        }
