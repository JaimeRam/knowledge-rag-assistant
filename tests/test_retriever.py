import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def mock_qdrant():
    with patch("app.rag.retriever.QdrantManager") as mock_cls:
        instance = MagicMock()
        mock_cls.return_value = instance
        yield instance


@pytest.fixture
def mock_embedder():
    with patch("app.rag.retriever.Embedder") as mock_cls:
        instance = MagicMock()
        mock_cls.return_value = instance
        yield instance


async def test_retrieve_returns_chunks(mock_qdrant, mock_embedder, sample_chunks):
    from app.rag.retriever import Retriever

    mock_embedder.embed_text = AsyncMock(return_value=[0.1] * 1024)
    mock_qdrant.search = AsyncMock(return_value=sample_chunks)

    retriever = Retriever()
    results = await retriever.retrieve("Tell me about Agumon")

    assert len(results) == 2
    assert results[0]["payload"]["name"] == "Agumon"


async def test_retrieve_empty_when_no_embedding(mock_qdrant, mock_embedder):
    from app.rag.retriever import Retriever

    mock_embedder.embed_text = AsyncMock(return_value=None)

    retriever = Retriever()
    results = await retriever.retrieve("anything")

    assert results == []
    mock_qdrant.search.assert_not_called()


async def test_retrieve_respects_limit(mock_qdrant, mock_embedder, sample_chunks):
    from app.rag.retriever import Retriever

    mock_embedder.embed_text = AsyncMock(return_value=[0.1] * 1024)
    mock_qdrant.search = AsyncMock(return_value=sample_chunks[:1])

    retriever = Retriever()
    results = await retriever.retrieve("query", limit=1)

    _, kwargs = mock_qdrant.search.call_args
    assert kwargs.get("limit") == 1


async def test_retrieve_passes_filters(mock_qdrant, mock_embedder, sample_chunks):
    from app.rag.retriever import Retriever

    mock_embedder.embed_text = AsyncMock(return_value=[0.1] * 1024)
    mock_qdrant.search = AsyncMock(return_value=sample_chunks)

    retriever = Retriever()
    await retriever.retrieve("rookie digimon", filters={"level": "Rookie"})

    _, kwargs = mock_qdrant.search.call_args
    assert kwargs.get("filter_params") == {"level": "Rookie"}
