import pytest
from app.ingestion.dapi_client import DAPIClient
from app.ingestion.embedder import Embedder


@pytest.mark.asyncio
async def test_dapi_client_fetch_list():
    """Test fetching digimon list from DAPI."""
    client = DAPIClient()
    digimons = await client.get_digimon_list(page=0, page_size=10)
    
    assert isinstance(digimons, list)
    assert len(digimons) <= 10
    
    if digimons:
        assert "id" in digimons[0]
        assert "name" in digimons[0]
    
    await client.close()


@pytest.mark.asyncio
async def test_embedder():
    """Test embedding generation."""
    embedder = Embedder()
    embedding = await embedder.embed_text("Agumon is a Rookie Digimon")
    
    assert isinstance(embedding, list)
    assert len(embedding) == 1536  # OpenAI text-embedding-3-small dimension
    
    await embedder.close()


@pytest.mark.asyncio
async def test_dapi_client_fetch_single():
    """Test fetching a single digimon."""
    client = DAPIClient()
    digimon = await client.get_digimon(1)
    
    if digimon:
        assert "id" in digimon
        assert "name" in digimon
    
    await client.close()
