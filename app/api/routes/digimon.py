from fastapi import APIRouter, HTTPException
from app.db.postgres import PostgresManager, DigimonMetadata
from sqlalchemy.orm import Session
from typing import List, Dict, Any

router = APIRouter(prefix="/api/v1", tags=["digimon"])


@router.get("/digimon/{digimon_id}")
async def get_digimon(digimon_id: int):
    """Get Digimon metadata by ID."""
    postgres = PostgresManager()
    session = postgres.get_session()
    
    try:
        digimon = session.query(DigimonMetadata).filter(
            DigimonMetadata.dapi_id == digimon_id
        ).first()
        
        if not digimon:
            raise HTTPException(status_code=404, detail="Digimon not found")
        
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
        postgres.close()


@router.get("/digimon")
async def list_digimons(limit: int = 10, offset: int = 0):
    """List Digimons with pagination."""
    postgres = PostgresManager()
    session = postgres.get_session()
    
    try:
        digimons = session.query(DigimonMetadata).offset(offset).limit(limit).all()
        
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
        postgres.close()
