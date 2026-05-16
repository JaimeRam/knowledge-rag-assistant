from app.agents.graph.state import GraphState
from app.rag.retriever import Retriever
from app.rag.llm_client import LLMClient
from app.rag.prompt_builder import PromptBuilder
import logging

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 2


async def retrieve_node(state: GraphState) -> dict:
    """Retrieve relevant chunks from Qdrant for the current query."""
    active_query = state.get("rewritten_query") or state["query"]
    retriever = Retriever()
    try:
        documents = await retriever.retrieve(active_query, limit=5)
        logger.info(f"Retrieved {len(documents)} documents for: {active_query!r}")
        return {"documents": documents}
    finally:
        await retriever.close()


async def grade_node(state: GraphState) -> dict:
    """Ask the LLM whether the retrieved documents are relevant to the query."""
    query = state["query"]
    documents = state["documents"]

    if not documents:
        return {"relevant": False}

    context = "\n".join(
        f"- {d.get('payload', {}).get('chunk_text', '')[:200]}" for d in documents
    )
    prompt = (
        f"Are the following documents relevant to answer the question: '{query}'?\n\n"
        f"Documents:\n{context}\n\n"
        "Reply with only YES or NO."
    )

    llm = LLMClient()
    try:
        response = await llm.chat(prompt, max_tokens=10)
        relevant = "yes" in response["content"].lower()
        logger.info(f"Grade result: {'relevant' if relevant else 'not relevant'}")
        return {"relevant": relevant}
    finally:
        await llm.close()


async def generate_node(state: GraphState) -> dict:
    """Generate an answer using the retrieved documents."""
    query = state["query"]
    documents = state["documents"]

    prompt = PromptBuilder.build_chat_prompt(query, documents)
    llm = LLMClient()
    try:
        response = await llm.chat(prompt)
        logger.info("Generated response")
        return {"generation": response["content"]}
    finally:
        await llm.close()


async def rewrite_node(state: GraphState) -> dict:
    """Rewrite the query to improve retrieval quality."""
    original = state["query"]
    iterations = state.get("iterations", 0) + 1

    prompt = (
        f"The following question did not yield relevant Digimon documents: '{original}'\n\n"
        "Rewrite it to be more specific and improve search results. "
        "Return only the rewritten question, nothing else."
    )

    llm = LLMClient()
    try:
        response = await llm.chat(prompt, max_tokens=100)
        rewritten = response["content"].strip()
        logger.info(f"Rewrote query to: {rewritten!r} (iteration {iterations})")
        return {"rewritten_query": rewritten, "iterations": iterations}
    finally:
        await llm.close()


def decide_to_generate(state: GraphState) -> str:
    """Route: generate if docs are relevant, rewrite if not (up to MAX_ITERATIONS)."""
    if state["relevant"]:
        return "generate"
    if state.get("iterations", 0) >= MAX_ITERATIONS:
        logger.warning("Max iterations reached, generating with available documents")
        return "generate"
    return "rewrite"
