from app.ingestion.dapi_client import DAPIClient
from app.db.postgres import PostgresManager, DigimonMetadata
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class MCPTools:
    """MCP tools for Digimon data access."""
    
    def __init__(self):
        self.dapi_client = DAPIClient()
        self.postgres = PostgresManager()
    
    async def get_digimon_by_name(self, name: str) -> Dict[str, Any]:
        """Get Digimon information by name from local database."""
        session = self.postgres.get_session()
        try:
            digimon = session.query(DigimonMetadata).filter(
                DigimonMetadata.name.ilike(f"%{name}%")
            ).first()
            
            if not digimon:
                return {"error": f"Digimon '{name}' not found"}
            
            return {
                "id": digimon.dapi_id,
                "name": digimon.name,
                "level": digimon.level,
                "type": digimon.type,
                "attribute": digimon.attribute,
                "description": digimon.description,
                "image_url": digimon.image_url
            }
        finally:
            session.close()
    
    async def get_digimon_by_level(self, level: str) -> List[Dict[str, Any]]:
        """Get Digimons by level from local database."""
        session = self.postgres.get_session()
        try:
            digimons = session.query(DigimonMetadata).filter(
                DigimonMetadata.level == level
            ).limit(20).all()
            
            return [
                {
                    "id": d.dapi_id,
                    "name": d.name,
                    "level": d.level,
                    "type": d.type
                }
                for d in digimons
            ]
        finally:
            session.close()
    
    async def get_digimon_by_id(self, digimon_id: int) -> Dict[str, Any]:
        """Get Digimon by DAPI ID."""
        digimon = await self.dapi_client.get_digimon(digimon_id)
        
        if not digimon:
            return {"error": f"Digimon with ID {digimon_id} not found"}
        
        return digimon
    
    async def get_digimon_skills(self, digimon_id: int) -> List[str]:
        """Get skills for a specific Digimon."""
        digimon = await self.dapi_client.get_digimon(digimon_id)
        
        if not digimon:
            return []
        
        skills = digimon.get("skills", [])
        return [s.get("skill", "") for s in skills if s.get("skill")]
    
    async def close(self):
        """Close resources."""
        await self.dapi_client.close()
        self.postgres.close()
