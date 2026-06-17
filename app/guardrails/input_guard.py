from app.rag.llm_client import LLMClient
import logging

logger = logging.getLogger(__name__)

_CLASSIFICATION_PROMPT = """\
You are a query classifier. Determine if the following question is related to Digimon \
(the anime/game franchise, its characters, evolutions, abilities, or lore).

Question: {query}

Reply with ONLY the word YES if it is Digimon-related, or ONLY the word NO if it is not."""

OFF_TOPIC_MESSAGE = (
    "I'm only able to answer questions about Digimon. "
    "Please ask me something about Digimon characters, evolutions, or abilities!"
)


class InputGuard:
    """Classify queries as Digimon-related before forwarding to the RAG pipeline."""

    def __init__(self, llm_client: LLMClient):
        self._llm = llm_client

    async def is_digimon_related(self, query: str) -> bool:
        """Return True if the query is about Digimon.

        Fail-open: returns True on any LLM error to avoid cascading failures.
        """
        try:
            result = await self._llm.chat(
                _CLASSIFICATION_PROMPT.format(query=query),
                max_tokens=10,
            )
            answer = result.get("content", "").strip().lower()
            return answer == "yes"
        except Exception as e:
            logger.warning(f"Guardrail classification failed (fail-open): {e}")
            return True
