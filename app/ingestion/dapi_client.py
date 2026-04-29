import httpx
from typing import Dict, Any, List, Optional
from app.core.config import get_settings
import logging

logger = logging.getLogger(__name__)


class DAPIClient:
    """Client for the Digimon API (DAPI)."""
    
    def __init__(self):
        settings = get_settings()
        self.base_url = settings.dapi_base_url
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def get_digimon_list(self, page: int = 0, page_size: int = 100) -> List[Dict[str, Any]]:
        """Get list of Digimons with pagination."""
        try:
            response = await self.client.get(
                f"{self.base_url}/digimon",
                params={"page": page, "pageSize": page_size}
            )
            response.raise_for_status()
            data = response.json()
            return data.get("content", [])
        except Exception as e:
            logger.error(f"Error fetching digimon list: {e}")
            return []
    
    async def get_digimon(self, digimon_id: int) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific Digimon."""
        try:
            response = await self.client.get(f"{self.base_url}/digimon/{digimon_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching digimon {digimon_id}: {e}")
            return None
    
    async def get_all_digimons(self) -> List[Dict[str, Any]]:
        """Fetch all Digimons from the API."""
        all_digimons = []
        page = 0
        
        while True:
            digimons = await self.get_digimon_list(page=page, page_size=100)
            if not digimons:
                break
            
            all_digimons.extend(digimons)
            page += 1
            logger.info(f"Fetched {len(all_digimons)} digimons so far...")
        
        logger.info(f"Total digimons fetched: {len(all_digimons)}")
        return all_digimons
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
