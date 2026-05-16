# Knowledge RAG Assistant

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A production-ready RAG-powered knowledge assistant demonstrating modern LLM backend development practices: custom RAG pipeline, MCP server, PydanticAI tool-calling agent, LangGraph Corrective RAG, Langfuse observability, and semantic caching.

*Example domain: Digimon knowledge base (DAPI)*

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI (Python 3.11+) |
| LLM | Anthropic `claude-haiku-4-5` |
| Embeddings | Voyage AI `voyage-3` (1024 dims) |
| Vector DB | Qdrant |
| Relational DB | PostgreSQL + SQLAlchemy |
| Caching | Redis |
| Observability | Langfuse |
| Agent Protocol | MCP (Model Context Protocol) |
| Typed Agent | PydanticAI (tool-calling agent) |
| Agentic RAG | LangGraph (Corrective RAG workflow) |
| Infrastructure | Docker + docker-compose |

## Architecture

```
  ┌─────────────────────────────────────────────────────┐
  │                    Client (HTTP)                    │
  └─────────────────────────┬───────────────────────────┘
                            │
             ┌──────────────┴─────────────┐
             │                            │
  ┌──────────▼──────────────┐  ┌──────────▼──────────────────────┐
  │   POST /api/v1/chat     │  │   POST /api/v1/agent            │
  │      (Custom RAG)       │  │   POST /api/v1/agent/graph      │
  └──────────┬──────────────┘  └──────────┬──────────────────────┘
             │                            │
             └──────────────┬─────────────┘
                            │
  ┌─────────────────────────▼───────────────────────────┐
  │                      RAG Core                       │
  │      Retriever → PromptBuilder → LLMClient          │
  └───────────────────┬─────────────────┬───────────────┘
                      │                 │
           ┌──────────▼──┐     ┌────────▼──────┐
           │   Qdrant    │     │     DAPI      │
           │  (vectors)  │     │  (live API)   │
           └─────────────┘     └───────────────┘

  LangGraph Corrective RAG flow:
  retrieve → grade → generate
                └──→ rewrite → retrieve (max 2 loops)

  Observability: Langfuse traces all LLM calls and retrievals
  Caching:       Redis caches /chat responses by query + filters
```

## Features

- **Custom RAG Pipeline**: Retriever → Prompt Builder → LLM, built from scratch without LangChain
- **PydanticAI Agent** (`/api/v1/agent`): type-safe agent with 4 tools (RAG search, by name, by level, skills) and automatic tool selection
- **LangGraph Corrective RAG** (`/api/v1/agent/graph`): agentic retrieve → grade → generate workflow with query rewriting when docs aren't relevant
- **MCP Server**: 4 tools exposing Digimon data (by name, level, ID, skills)
- **Observability**: Full tracing with Langfuse (LLM calls, retrievals, tool invocations)
- **Semantic Caching**: Redis-based response caching to reduce API costs
- **Data Ingestion**: Async pipeline from [DAPI](https://digi-api.com/) → Voyage AI embeddings → Qdrant + PostgreSQL

## Setup

### Prerequisites

- Docker and Docker Compose
- Python 3.11+
- Anthropic API key → [console.anthropic.com](https://console.anthropic.com)
- Voyage AI API key → [dash.voyageai.com](https://dash.voyageai.com)

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
# Edit .env — set ANTHROPIC_API_KEY and VOYAGE_API_KEY at minimum
```

5. Start infrastructure services (Qdrant, Redis, PostgreSQL, Langfuse):

```bash
make services
```

6. Run data ingestion:

```bash
make ingest
```

> **Voyage AI free tier note**: without a payment method the limit is 3 RPM. The default `INGEST_LIMIT=100` in `.env` keeps ingestion under ~5 minutes. Set `INGEST_LIMIT=0` to ingest all ~1450 Digimon once you add a payment method (200M free tokens still apply).

7. Start the API:

```bash
make run
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

### PydanticAI Agent (tool-calling)

```bash
POST /api/v1/agent
Content-Type: application/json

{"query": "What are the skills of Agumon?"}
# → agent selects get_digimon_by_name + get_digimon_skills automatically

{"query": "Which Rookie Digimon are best for beginners?"}
# → agent selects rag_search + get_digimon_by_level automatically
```

Response includes `tool_calls` (list of tools used) and token `usage`.

### LangGraph Corrective RAG

```bash
POST /api/v1/agent/graph
Content-Type: application/json

{"query": "Tell me about fire-type Rookie Digimon"}
```

Response includes `iterations` (0 = direct, 1+ = query was rewritten) and `documents_used`.

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

1. Fetches Digimon data (respects `INGEST_LIMIT` in `.env`)
2. Prepares text chunks per Digimon (basic info, description, skills)
3. Generates vector embeddings with Voyage AI (`voyage-3`, 1024 dims) in rate-limited batches
4. Stores embeddings in Qdrant and metadata in PostgreSQL

Key `.env` variables for ingestion:

| Variable | Default | Description |
|---|---|---|
| `INGEST_LIMIT` | `100` | Max Digimon to ingest (0 = all ~1450) |
| `VOYAGE_EMBED_BATCH_SIZE` | `20` | Chunks per embedding API call |
| `VOYAGE_RPM` | `3` | Voyage AI requests per minute (free tier = 3) |

To re-run ingestion:

```bash
make ingest
```

## Project Structure

```
knowledge-rag-assistant/
├── app/
│   ├── api/              # FastAPI routes and app setup
│   │   ├── routes/
│   │   │   ├── chat.py       # /chat — Custom RAG + Redis caching
│   │   │   ├── agent.py      # /agent — PydanticAI + /agent/graph — LangGraph
│   │   │   ├── digimon.py    # Digimon metadata endpoints
│   │   │   └── health.py     # Health check
│   │   └── main.py
│   ├── core/             # Config (pydantic-settings) and logging
│   ├── ingestion/        # DAPI client, embedder, ingest pipeline
│   ├── rag/              # Retriever, prompt builder, LLM client
│   ├── mcp/              # MCP server with 4 Digimon tools
│   ├── agents/
│   │   ├── pydantic_agent.py # PydanticAI agent with 4 typed tools
│   │   └── graph/            # LangGraph Corrective RAG
│   │       ├── state.py      # GraphState TypedDict
│   │       ├── nodes.py      # retrieve / grade / generate / rewrite nodes
│   │       └── workflow.py   # Compiled StateGraph
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

1. **Custom RAG without LangChain**: Built the full pipeline (retrieval → prompt → generation) from first principles — shows deep understanding of how RAG works, not just framework usage.

2. **Three agent patterns side by side**: Custom RAG (`/chat`), PydanticAI typed agent (`/agent`), and LangGraph Corrective RAG (`/agent/graph`) — each demonstrates a different architectural trade-off.

3. **PydanticAI typed tools**: Type-safe agent with `@agent.tool_plain` decorators, `AnthropicModel`, and automatic tool selection. Zero boilerplate.

4. **LangGraph Corrective RAG**: Stateful `retrieve → grade → generate` loop with query rewriting when retrieved docs aren't relevant — demonstrates iterative reasoning beyond single-shot generation.

5. **MCP (Model Context Protocol)**: Implements the emerging standard for tool integration between LLMs and external systems.

6. **Production Observability**: Full tracing with Langfuse covering LLM calls, retrieval steps, and tool invocations — critical for debugging and cost monitoring.

7. **Cost Optimization**: Redis caching avoids redundant LLM calls for identical queries, with proper cache-key design to prevent collisions across different filter combinations.

## Roadmap

- [x] Custom RAG pipeline (Retriever → PromptBuilder → LLMClient)
- [x] MCP server with 4 tools
- [x] Anthropic Claude Haiku + Voyage AI voyage-3 integration
- [x] Langfuse observability
- [x] Redis caching with proper cache-key design
- [x] Docker infrastructure
- [x] PydanticAI typed agent with automatic tool selection
- [x] LangGraph Corrective RAG with query rewriting
- [ ] Comprehensive test suite with pytest
- [ ] RAG evaluation pipeline (precision, recall, faithfulness)
- [ ] Guardrails for output validation
- [ ] Rate limiting and authentication

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

MIT — see [LICENSE](LICENSE) for details.
