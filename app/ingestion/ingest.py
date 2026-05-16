import asyncio
import math
from typing import List, Dict, Any, Tuple, Optional
from qdrant_client.models import PointStruct
from sqlalchemy.dialects.postgresql import insert as pg_insert
from app.ingestion.dapi_client import DAPIClient
from app.ingestion.embedder import Embedder
from app.db.qdrant import QdrantManager
from app.db.postgres import PostgresManager, DigimonMetadata
from app.core.config import get_settings
import logging

logger = logging.getLogger(__name__)


class DataIngestor:
    """Ingest Digimon data from DAPI into Qdrant and PostgreSQL."""

    def __init__(self):
        settings = get_settings()
        self.dapi_client = DAPIClient()
        self.embedder = Embedder()
        self.qdrant = QdrantManager()
        self.postgres = PostgresManager()
        self.limit = settings.ingest_limit
        self.batch_size = settings.voyage_embed_batch_size
        self.rpm = settings.voyage_rpm

    def _extract_description(self, digimon: Dict[str, Any]) -> str:
        """Extract English description from the individual DAPI endpoint format."""
        descriptions = digimon.get("descriptions", [])
        if descriptions:
            en = next((d.get("description", "") for d in descriptions if d.get("language") == "en_us"), "")
            return en or descriptions[0].get("description", "")
        return digimon.get("description", "")

    def _prepare_text_chunks(self, digimon: Dict[str, Any]) -> List[str]:
        """Prepare text chunks for embedding from a single Digimon."""
        chunks = []
        name = digimon.get("name", "")
        level = digimon.get("levels", [{}])[0].get("level", "") if digimon.get("levels") else ""
        digimon_type = digimon.get("types", [{}])[0].get("type", "") if digimon.get("types") else ""
        attribute = digimon.get("attributes", [{}])[0].get("attribute", "") if digimon.get("attributes") else ""
        description = self._extract_description(digimon)

        chunks.append(f"{name} is a {level} level Digimon with {digimon_type} type and {attribute} attribute.")

        if description:
            chunks.append(f"{name}: {description}")

        skills = digimon.get("skills", [])
        skill_names = [s.get("skill", "") for s in skills if s.get("skill")]
        if skill_names:
            chunks.append(f"{name} has the following skills: {', '.join(skill_names)}")

        return chunks

    async def _fetch_details(self, basic_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fetch full Digimon details concurrently (max 10 in-flight at a time)."""
        semaphore = asyncio.Semaphore(10)

        async def fetch_one(basic: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            async with semaphore:
                return await self.dapi_client.get_digimon(basic["id"])

        results = await asyncio.gather(*[fetch_one(b) for b in basic_list], return_exceptions=True)
        full = [r for r in results if r and not isinstance(r, Exception)]
        logger.info(f"Fetched full details for {len(full)}/{len(basic_list)} Digimon")
        return full

    async def _embed_all(self, texts: List[str]) -> List[List[float]]:
        """Embed all texts in rate-limited batches (free Voyage AI tier: 3 RPM)."""
        delay = 60.0 / self.rpm
        total_batches = math.ceil(len(texts) / self.batch_size)
        estimated_minutes = (total_batches * delay) / 60

        logger.info(
            f"Embedding {len(texts)} chunks in {total_batches} batches "
            f"({self.batch_size} chunks/batch, {self.rpm} RPM). "
            f"Estimated time: ~{estimated_minutes:.1f} min"
        )

        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch_num = i // self.batch_size + 1
            batch = texts[i : i + self.batch_size]

            if i > 0:
                logger.info(f"Rate limit pause ({delay:.0f}s)...")
                await asyncio.sleep(delay)

            logger.info(f"Batch {batch_num}/{total_batches} — {len(batch)} chunks")
            embeddings = await self.embedder.embed_texts(batch, input_type="document")

            if not embeddings:
                logger.error(f"Batch {batch_num} returned no embeddings — aborting")
                raise RuntimeError("Embedding API returned empty response. Check rate limits.")

            all_embeddings.extend(embeddings)

        return all_embeddings

    async def ingest(self):
        """Main ingestion pipeline."""
        logger.info("Starting data ingestion...")

        await self.qdrant.create_collection()
        self.postgres.create_tables()

        # Phase 1: Fetch basic list, then full details for each Digimon
        basic_list = await self.dapi_client.get_all_digimons()
        total_available = len(basic_list)
        if self.limit > 0:
            logger.info(f"INGEST_LIMIT={self.limit} — processing {self.limit} of {total_available} Digimon")
            basic_list = basic_list[: self.limit]

        digimons = await self._fetch_details(basic_list)
        logger.info(f"Ready to ingest {len(digimons)} Digimon")

        # Phase 2: Collect all text chunks
        all_texts: List[str] = []
        chunk_meta: List[Tuple[Dict[str, Any], int]] = []

        for digimon in digimons:
            chunks = self._prepare_text_chunks(digimon)
            for j, chunk in enumerate(chunks):
                all_texts.append(chunk)
                chunk_meta.append((digimon, j))

        logger.info(f"Prepared {len(all_texts)} chunks from {len(digimons)} Digimon")

        # Phase 3: Embed all chunks in rate-limited batches
        all_embeddings = await self._embed_all(all_texts)

        # Phase 4: Build Qdrant points + PostgreSQL upsert records
        session = self.postgres.get_session()
        points: List[PointStruct] = []
        pg_records: List[Dict[str, Any]] = []
        seen_ids: set = set()

        for (text, (digimon, j)), embedding in zip(zip(all_texts, chunk_meta), all_embeddings):
            digimon_id = digimon.get("id")
            name = digimon.get("name")
            level = digimon.get("levels", [{}])[0].get("level", "") if digimon.get("levels") else ""
            digimon_type = digimon.get("types", [{}])[0].get("type", "") if digimon.get("types") else ""
            attribute = digimon.get("attributes", [{}])[0].get("attribute", "") if digimon.get("attributes") else ""

            point_id = digimon_id * 10 + j
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "digimon_id": digimon_id,
                        "name": name,
                        "chunk_index": j,
                        "chunk_text": text,
                        "level": level,
                        "type": digimon_type,
                        "attribute": attribute,
                    },
                )
            )

            if digimon_id not in seen_ids:
                seen_ids.add(digimon_id)
                pg_records.append({
                    "dapi_id": digimon_id,
                    "name": name,
                    "level": level or None,
                    "type": digimon_type or None,
                    "attribute": attribute or None,
                    "description": self._extract_description(digimon) or None,
                    "image_url": digimon.get("images", [{}])[0].get("href") if digimon.get("images") else None,
                })

        # Phase 5: Persist — Qdrant upsert + PostgreSQL upsert (safe to re-run)
        for i in range(0, len(points), 100):
            await self.qdrant.upsert_points(points[i : i + 100])

        if pg_records:
            stmt = pg_insert(DigimonMetadata).values(pg_records)
            stmt = stmt.on_conflict_do_update(
                index_elements=["dapi_id"],
                set_={
                    "name": stmt.excluded.name,
                    "level": stmt.excluded.level,
                    "type": stmt.excluded.type,
                    "attribute": stmt.excluded.attribute,
                    "description": stmt.excluded.description,
                    "image_url": stmt.excluded.image_url,
                },
            )
            session.execute(stmt)

        session.commit()
        session.close()

        await self.dapi_client.close()
        logger.info(
            f"Ingestion complete — {len(digimons)} Digimon, {len(all_texts)} chunks stored. "
            f"(Set INGEST_LIMIT=0 in .env to ingest all {total_available} Digimon)"
        )


async def main():
    ingestor = DataIngestor()
    await ingestor.ingest()


if __name__ == "__main__":
    asyncio.run(main())
