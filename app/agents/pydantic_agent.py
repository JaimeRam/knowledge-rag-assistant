from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from app.core.config import get_settings
import logging

logger = logging.getLogger(__name__)

settings = get_settings()
_provider = AnthropicProvider(api_key=settings.anthropic_api_key)
_model = AnthropicModel(settings.anthropic_model, provider=_provider)

agent = Agent(
    model=_model,
    system_prompt=(
        "You are a knowledgeable Digimon expert assistant. "
        "Help users learn about Digimons, their abilities, evolution paths, and characteristics. "
        "Use the available tools to look up accurate information before answering. "
        "Be friendly and informative. If you can't find the information, say so politely."
    ),
)


@agent.tool_plain
async def rag_search(query: str) -> str:
    """Search the Digimon knowledge base using semantic RAG retrieval. Use for general questions."""
    from app.rag.retriever import Retriever

    retriever = Retriever()
    try:
        chunks = await retriever.retrieve(query, limit=5)
        if not chunks:
            return "No relevant information found in the knowledge base."

        parts = []
        for i, chunk in enumerate(chunks, 1):
            chunk_text = chunk.get("payload", {}).get("chunk_text", "")
            name = chunk.get("payload", {}).get("name", "Unknown")
            score = chunk.get("score", 0)
            parts.append(f"{i}. {name} (relevance: {score:.2f}): {chunk_text}")

        return "\n".join(parts)
    finally:
        await retriever.close()


@agent.tool_plain
async def get_digimon_by_name(name: str) -> str:
    """Look up a specific Digimon by name to get its stats and description."""
    from app.mcp.tools import MCPTools

    mcp = MCPTools()
    try:
        result = await mcp.get_digimon_by_name(name)
        if "error" in result:
            return result["error"]
        return (
            f"ID: {result['id']}\n"
            f"Name: {result['name']}\n"
            f"Level: {result.get('level', 'Unknown')}\n"
            f"Type: {result.get('type', 'Unknown')}\n"
            f"Attribute: {result.get('attribute', 'Unknown')}\n"
            f"Description: {result.get('description', 'No description available')}"
        )
    finally:
        await mcp.close()


@agent.tool_plain
async def get_digimon_by_level(level: str) -> str:
    """List Digimon by evolution level. Valid levels: Fresh, In-Training, Rookie, Champion, Ultimate, Mega."""
    from app.mcp.tools import MCPTools

    mcp = MCPTools()
    try:
        results = await mcp.get_digimon_by_level(level)
        if not results:
            return f"No Digimon found at level '{level}'."
        lines = [f"- {d['name']} (Type: {d.get('type', 'Unknown')})" for d in results]
        return f"{level} Digimon:\n" + "\n".join(lines)
    finally:
        await mcp.close()


@agent.tool_plain
async def get_digimon_skills(digimon_id: int) -> str:
    """Get the skills/attacks of a Digimon by its numeric DAPI ID."""
    from app.mcp.tools import MCPTools

    mcp = MCPTools()
    try:
        skills = await mcp.get_digimon_skills(digimon_id)
        if not skills:
            return f"No skills found for Digimon ID {digimon_id}."
        return f"Skills: {', '.join(skills)}"
    finally:
        await mcp.close()
