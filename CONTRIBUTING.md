# Contributing to Digimon RAG Assistant

Thank you for your interest in contributing to this project!

## Development Setup

1. Clone the repository
2. Create a virtual environment: `python3 -m venv venv`
3. Activate the virtual environment: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Copy `.env.example` to `.env` and add your OpenAI API key
6. Start Docker services: `cd docker && docker-compose up -d`
7. Run data ingestion: `python -m app.ingestion.ingest`
8. Start the server: `python -m app.api.main`

## Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to functions and classes
- Keep functions focused and small

## Testing

Run tests with: `pytest`

## Submitting Changes

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## Project Structure

```
app/
├── api/           # FastAPI routes
├── core/          # Configuration and logging
├── ingestion/     # Data ingestion from DAPI
├── rag/           # RAG pipeline
├── mcp/           # MCP server
├── agents/        # Agent SDK integration
├── observability/ # Langfuse integration
└── db/            # Database managers
```

## Questions?

Feel free to open an issue for any questions or suggestions.
