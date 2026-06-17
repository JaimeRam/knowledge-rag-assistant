from fastapi import APIRouter, HTTPException, Query
from sqlalchemy.exc import SQLAlchemyError
from app.db.postgres import PostgresManager, DigimonMetadata
import logging

logger = logging.getLogger(__name__)

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
            "image_url": digimon.image_url,
        }

    except HTTPException:
        raise
    except SQLAlchemyError as e:
        logger.error(f"Database error fetching digimon {digimon_id}: {e}")
        raise HTTPException(status_code=503, detail="Database unavailable")

    finally:
        session.close()
        postgres.close()


@router.get("/digimon")
async def list_digimons(
    limit: int = Query(default=10, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
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
                "type": d.type,
            }
            for d in digimons
        ]

    except SQLAlchemyError as e:
        logger.error(f"Database error listing digimons: {e}")
        raise HTTPException(status_code=503, detail="Database unavailable")

    finally:
        session.close()
        postgres.close()
