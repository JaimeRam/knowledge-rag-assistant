# Knowledge RAG Assistant

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A production-ready RAG-powered knowledge assistant demonstrating modern LLM backend development practices: custom RAG pipeline, MCP server, Agent SDK orchestration, Langfuse observability, and semantic caching. 

*Example domain: Digimon knowledge base (DAPI)*

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI (Python 3.11+) |
| LLM | OpenAI `gpt-4o-mini` + `text-embedding-3-small` |
| Vector DB | Qdrant |
| Relational DB | PostgreSQL + SQLAlchemy |
| Caching | Redis |
| Observability | Langfuse |
| Agent Protocol | MCP (Model Context Protocol) |
| Agent SDK | OpenAI Agents SDK |
| Infrastructure | Docker + docker-compose |

## Architecture

```
┌──────────────┐
│    Client    │  (HTTP)
└───────┬──────┘
        │
┌───────▼──────┐
│   FastAPI    │
└───────┬──────┘
        │
┌───────▼──────────────────┐
│  Agent Orchestrator      │
│  (OpenAI Agents SDK)     │
└───────┬──────────────────┘
        │
   ┌────┴────┐
   │         │
┌──▼───┐  ┌──▼───┐
│ RAG  │  │ MCP  │
└──┬───┘  └──┬───┘
   │         │
┌──▼───┐  ┌──▼───┐
│Qdrant│  │ DAPI │
└──────┘  └──────┘

Observability: Langfuse traces all LLM calls and retrievals
Caching: Redis caches responses by query similarity
```

## Features

- **Custom RAG Pipeline**: Retriever → Prompt Builder → LLM, built from scratch without LangChain
- **MCP Server**: 4 tools exposing Digimon data (by name, level, ID, skills)
- **Agent Orchestration**: Multi-step workflows with OpenAI Agents SDK and tool calling
- **Observability**: Full tracing with Langfuse (LLM calls, retrievals, tool invocations)
- **Semantic Caching**: Redis-based response caching to reduce OpenAI API costs
- **Data Ingestion**: Async pipeline from [DAPI](https://digi-api.com/) → embeddings → Qdrant + PostgreSQL

## Setup

### Prerequisites

- Docker and Docker Compose
- Python 3.11+
- OpenAI API key (with available credits)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/JaimeRam/knowledge-rag-assistant.git
cd knowledge-rag-assistant
```

2. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install Python dependencies:

```bash
pip install -r requirements.txt
```

4. Copy and configure the environment file:

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

5. Start infrastructure services:

```bash
cd docker && docker-compose up -d
```

6. Run data ingestion:

```bash
python -m app.ingestion.ingest
```

7. Start the FastAPI application:

```bash
python -m app.api.main
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### Docker (full stack)

```bash
docker build -t digimon-rag-assistant .
docker run -p 8000:8000 --env-file .env digimon-rag-assistant
```

## API Endpoints

### Chat with RAG

```bash
POST /api/v1/chat
Content-Type: application/json

{
  "query": "What is Agumon?",
  "level_filter": "Rookie",
  "type_filter": "Reptile"
}
```

### Get Digimon by ID

```bash
GET /api/v1/digimon/{id}
```

### List Digimons

```bash
GET /api/v1/digimon?limit=10&offset=0
```

### Health Check

```bash
GET /api/v1/health
```

## Data Ingestion

The ingestion pipeline fetches data from [DAPI](https://digi-api.com/):

1. Fetches all Digimons from the REST API
2. Chunks and processes each Digimon's information
3. Generates vector embeddings with OpenAI
4. Stores embeddings in Qdrant and metadata in PostgreSQL

To re-run ingestion:

```bash
python -m app.ingestion.ingest
```

## Project Structure

```
knowledge-rag-assistant/
├── app/
│   ├── api/              # FastAPI routes and app setup
│   │   ├── routes/
│   │   │   ├── chat.py       # Chat endpoint with RAG + caching
│   │   │   ├── digimon.py    # Digimon metadata endpoints
│   │   │   └── health.py     # Health check
│   │   └── main.py
│   ├── core/             # Config (pydantic-settings) and logging
│   ├── ingestion/        # DAPI client, embedder, ingest pipeline
│   ├── rag/              # Retriever, prompt builder, LLM client
│   ├── mcp/              # MCP server with 4 Digimon tools
│   ├── agents/           # OpenAI Agents SDK orchestrator
│   ├── observability/    # Langfuse tracing integration
│   └── db/               # Qdrant, PostgreSQL, Redis managers
├── docker/
│   └── docker-compose.yml
├── tests/
├── requirements.txt
├── Dockerfile
├── CONTRIBUTING.md
└── README.md
```

## Technical Highlights

This project demonstrates key skills for LLM Backend Developer roles:

1. **Custom RAG without LangChain**: Built the full pipeline (retrieval → prompt → generation) from first principles, showing deep understanding of how RAG works.

2. **MCP (Model Context Protocol)**: Implements the emerging standard for tool integration between LLMs and external systems.

3. **Agent Orchestration**: Uses OpenAI Agents SDK for multi-step agentic workflows with structured tool calling.

4. **Production Observability**: Full tracing with Langfuse covering LLM calls, retrieval steps, and tool invocations — critical for debugging and cost monitoring.

5. **Cost Optimization**: Semantic caching with Redis avoids redundant LLM calls for similar queries.

6. **Production-Ready Patterns**: Docker deployment, async FastAPI, pydantic-settings config, structured logging, and proper error handling.

## Roadmap

- [x] Custom RAG pipeline
- [x] MCP server with 4 tools
- [x] OpenAI Agents SDK integration
- [x] Langfuse observability
- [x] Redis semantic caching
- [x] Docker infrastructure
- [ ] Comprehensive test suite with pytest
- [ ] RAG evaluation pipeline (precision, recall, faithfulness)
- [ ] Guardrails for output validation
- [ ] Rate limiting and authentication

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

MIT — see [LICENSE](LICENSE) for details.
