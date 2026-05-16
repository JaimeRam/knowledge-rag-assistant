from langfuse import Langfuse
from app.core.config import get_settings
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class LangfuseManager:
    """Manage Langfuse observability integration (SDK v2, server v2.x)."""

    def __init__(self):
        settings = get_settings()

        if settings.langfuse_public_key and settings.langfuse_secret_key:
            self.client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
            self.enabled = True
            logger.info("Langfuse observability enabled")
        else:
            self.client = None
            self.enabled = False
            logger.warning("Langfuse observability disabled (missing credentials)")

    async def trace_chat(
        self,
        query: str,
        response: str,
        context_chunks: list,
        token_usage: Dict[str, int],
        latency_ms: int,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Trace a full chat interaction with its LLM generation."""
        if not self.enabled:
            return
        try:
            trace = self.client.trace(
                name="chat_interaction",
                input={"query": query},
                output={"response": response},
                metadata={
                    "context_sources": len(context_chunks),
                    "latency_ms": latency_ms,
                    **(metadata or {}),
                },
            )
            trace.generation(
                name="llm_call",
                input=query,
                output=response,
                usage={
                    "input": token_usage.get("prompt_tokens", 0),
                    "output": token_usage.get("completion_tokens", 0),
                    "total": token_usage.get("total_tokens", 0),
                },
            )
            logger.info(f"Traced chat interaction: {query[:50]}...")
        except Exception as e:
            logger.error(f"Error tracing chat interaction: {e}")

    async def trace_tool_call(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        result: Any,
        latency_ms: int,
    ):
        """Trace an MCP tool call."""
        if not self.enabled:
            return
        try:
            self.client.trace(
                name="tool_call",
                input={"tool": tool_name, "parameters": parameters},
                output={"result": str(result)[:500]},
                metadata={"latency_ms": latency_ms},
            )
            logger.info(f"Traced tool call: {tool_name}")
        except Exception as e:
            logger.error(f"Error tracing tool call: {e}")

    async def trace_retrieval(
        self,
        query: str,
        retrieved_count: int,
        avg_score: float,
        latency_ms: int,
    ):
        """Trace a vector retrieval operation."""
        if not self.enabled:
            return
        try:
            self.client.trace(
                name="retrieval",
                input={"query": query},
                output={"retrieved_count": retrieved_count, "avg_score": avg_score},
                metadata={"latency_ms": latency_ms},
            )
            logger.info(f"Traced retrieval: {retrieved_count} chunks, avg_score={avg_score:.3f}")
        except Exception as e:
            logger.error(f"Error tracing retrieval: {e}")

    def flush(self):
        """Flush buffered traces to the Langfuse server."""
        if self.enabled and self.client:
            try:
                self.client.flush()
            except Exception as e:
                logger.error(f"Error flushing Langfuse: {e}")
