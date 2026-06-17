import redis.asyncio as redis
import json
from typing import Optional, Any
from app.core.config import get_settings
import logging

logger = logging.getLogger(__name__)


class RedisManager:
    def __init__(self):
        settings = get_settings()
        self.client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            decode_responses=True
        )

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            value = await self.client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: int = 86400) -> bool:
        """Set value in cache with TTL (default 24h)."""
        try:
            await self.client.setex(key, ttl, json.dumps(value))
            return True
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        try:
            await self.client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        try:
            return await self.client.exists(key) > 0
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            return False

    async def close(self) -> None:
        """Close the Redis connection pool."""
        await self.client.aclose()

    async def flush_db(self) -> bool:
        """Flush the current database."""
        try:
            await self.client.flushdb()
            logger.info("Flushed Redis database")
            return True
        except Exception as e:
            logger.error(f"Redis flush error: {e}")
            return False
