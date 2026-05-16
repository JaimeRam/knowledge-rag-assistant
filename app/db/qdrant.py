from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
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
        self.vector_size = 1024  # Voyage AI voyage-3

    async def create_collection(self) -> None:
        """Create collection, recreating it if the vector dimension has changed."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name in collection_names:
            info = self.client.get_collection(self.collection_name)
            existing_size = info.config.params.vectors.size
            if existing_size != self.vector_size:
                logger.warning(
                    f"Collection '{self.collection_name}' has dim={existing_size}, "
                    f"expected {self.vector_size}. Recreating..."
                )
                self.client.delete_collection(self.collection_name)
            else:
                logger.info(f"Collection already exists: {self.collection_name}")
                return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
        )
        logger.info(f"Created collection: {self.collection_name} (dim={self.vector_size})")

    async def upsert_points(self, points: List[PointStruct]) -> None:
        """Upsert points to the collection."""
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        logger.info(f"Upserted {len(points)} points")

    def _build_filter(self, filter_params: Dict[str, Any]) -> Optional[Filter]:
        """Convert a plain dict of field=value pairs into a Qdrant Filter."""
        if not filter_params:
            return None
        conditions = [
            FieldCondition(key=key, match=MatchValue(value=value))
            for key, value in filter_params.items()
            if value is not None
        ]
        return Filter(must=conditions) if conditions else None

    async def search(
        self,
        query_vector: List[float],
        limit: int = 5,
        score_threshold: float = 0.5,
        filter_params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar points using query_points (qdrant-client ≥1.7)."""
        qdrant_filter = self._build_filter(filter_params or {})

        result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=qdrant_filter,
        )

        return [
            {"id": hit.id, "score": hit.score, "payload": hit.payload}
            for hit in result.points
        ]

    async def delete_collection(self) -> None:
        """Delete the collection."""
        self.client.delete_collection(self.collection_name)
        logger.info(f"Deleted collection: {self.collection_name}")

    async def get_collection_info(self) -> Dict[str, Any]:
        """Get collection info."""
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "points_count": info.points_count,
            "vector_size": info.config.params.vectors.size,
        }
