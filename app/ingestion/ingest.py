import asyncio
from typing import List, Dict, Any
from qdrant_client.models import PointStruct
from app.ingestion.dapi_client import DAPIClient
from app.ingestion.embedder import Embedder
from app.db.qdrant import QdrantManager
from app.db.postgres import PostgresManager, DigimonMetadata
import logging

logger = logging.getLogger(__name__)


class DataIngestor:
    """Ingest Digimon data from DAPI into Qdrant and PostgreSQL."""
    
    def __init__(self):
        self.dapi_client = DAPIClient()
        self.embedder = Embedder()
        self.qdrant = QdrantManager()
        self.postgres = PostgresManager()
    
    async def prepare_text_chunks(self, digimon: Dict[str, Any]) -> List[str]:
        """Prepare text chunks for embedding from Digimon data."""
        chunks = []
        
        name = digimon.get("name", "")
        level = digimon.get("levels", [{}])[0].get("level", "") if digimon.get("levels") else ""
        digimon_type = digimon.get("types", [{}])[0].get("type", "") if digimon.get("types") else ""
        attribute = digimon.get("attributes", [{}])[0].get("attribute", "") if digimon.get("attributes") else ""
        description = digimon.get("description", "")
        
        # Chunk 1: Basic info
        chunk1 = f"{name} is a Digimon of {level} level with {digimon_type} type and {attribute} attribute."
        chunks.append(chunk1)
        
        # Chunk 2: Description
        if description:
            chunk2 = f"{name}: {description}"
            chunks.append(chunk2)
        
        # Chunk 3: Skills
        skills = digimon.get("skills", [])
        if skills:
            skill_names = [s.get("name", "") for s in skills if s.get("name")]
            if skill_names:
                chunk3 = f"{name} has the following skills: {', '.join(skill_names)}"
                chunks.append(chunk3)
        
        return chunks
    
    async def ingest(self):
        """Main ingestion pipeline."""
        logger.info("Starting data ingestion...")
        
        # Initialize databases
        await self.qdrant.create_collection()
        self.postgres.create_tables()
        
        # Fetch all Digimons
        digimons = await self.dapi_client.get_all_digimons()
        logger.info(f"Fetched {len(digimons)} Digimons from DAPI")
        
        # Process each Digimon
        points = []
        session = self.postgres.get_session()
        
        for i, digimon in enumerate(digimons):
            digimon_id = digimon.get("id")
            name = digimon.get("name")
            
            logger.info(f"Processing {i+1}/{len(digimons)}: {name}")
            
            # Prepare text chunks
            chunks = await self.prepare_text_chunks(digimon)
            
            # Generate embeddings for chunks
            embeddings = await self.embedder.embed_texts(chunks)
            
            # Create points for Qdrant
            for j, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point_id = f"{digimon_id}_{j}"
                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "digimon_id": digimon_id,
                        "name": name,
                        "chunk_index": j,
                        "chunk_text": chunk,
                        "level": digimon.get("levels", [{}])[0].get("level", "") if digimon.get("levels") else "",
                        "type": digimon.get("types", [{}])[0].get("type", "") if digimon.get("types") else "",
                        "attribute": digimon.get("attributes", [{}])[0].get("attribute", "") if digimon.get("attributes") else ""
                    }
                )
                points.append(point)
            
            # Store metadata in PostgreSQL
            metadata = DigimonMetadata(
                dapi_id=digimon_id,
                name=name,
                level=digimon.get("levels", [{}])[0].get("level", "") if digimon.get("levels") else None,
                type=digimon.get("types", [{}])[0].get("type", "") if digimon.get("types") else None,
                attribute=digimon.get("attributes", [{}])[0].get("attribute", "") if digimon.get("attributes") else None,
                description=digimon.get("description"),
                image_url=digimon.get("images", [{}])[0].get("href", "") if digimon.get("images") else None
            )
            session.add(metadata)
            
            # Batch upsert every 100 points
            if len(points) >= 100:
                await self.qdrant.upsert_points(points)
                session.commit()
                points = []
                logger.info(f"Batch upsert completed. Total processed: {i+1}")
        
        # Final batch
        if points:
            await self.qdrant.upsert_points(points)
        
        session.commit()
        session.close()
        
        # Cleanup
        await self.dapi_client.close()
        await self.embedder.close()
        
        logger.info("Data ingestion completed successfully!")


async def main():
    """Run ingestion."""
    ingestor = DataIngestor()
    await ingestor.ingest()


if __name__ == "__main__":
    asyncio.run(main())
