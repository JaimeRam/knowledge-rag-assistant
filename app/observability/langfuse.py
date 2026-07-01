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

    def trace_chat(
        self,
        query: str,
        response: str,
        context_chunks: list,
        token_usage: Dict[str, int],
        latency_ms: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Trace a full chat interaction. Returns the trace ID (or None if disabled)."""
        if not self.enabled:
            return None
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

            # System metric scores (visible in Langfuse dashboard per trace)
            self.client.score(trace_id=trace.id, name="latency_ms", value=float(latency_ms))
            total_tokens = token_usage.get("total_tokens", 0)
            if total_tokens:
                self.client.score(
                    trace_id=trace.id, name="total_tokens", value=float(total_tokens)
                )
            if context_chunks:
                self.client.score(
                    trace_id=trace.id,
                    name="context_count",
                    value=float(len(context_chunks)),
                )
                avg = sum(c.get("score", 0) for c in context_chunks) / len(context_chunks)
                if avg > 0:
                    self.client.score(
                        trace_id=trace.id, name="retrieval_avg_score", value=avg
                    )

            logger.info(f"Traced chat interaction: {query[:50]}...")
            return trace.id
        except Exception as e:
            logger.error(f"Error tracing chat interaction: {e}")
            return None

    def trace_tool_call(
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

    def trace_retrieval(
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
            trace = self.client.trace(
                name="retrieval",
                input={"query": query},
                output={"retrieved_count": retrieved_count, "avg_score": avg_score},
                metadata={"latency_ms": latency_ms},
            )
            self.client.score(
                trace_id=trace.id, name="retrieved_count", value=float(retrieved_count)
            )
            if avg_score > 0:
                self.client.score(trace_id=trace.id, name="avg_score", value=avg_score)
            logger.info(f"Traced retrieval: {retrieved_count} chunks, avg_score={avg_score:.3f}")
        except Exception as e:
            logger.error(f"Error tracing retrieval: {e}")

    def add_score(
        self,
        trace_id: str,
        name: str,
        value: float,
        comment: Optional[str] = None,
    ):
        """Add a numeric score to an existing trace (e.g. LLM-as-judge results)."""
        if not self.enabled or not trace_id:
            return
        try:
            self.client.score(trace_id=trace_id, name=name, value=value, comment=comment)
        except Exception as e:
            logger.error(f"Error adding score '{name}' to trace {trace_id}: {e}")

    def flush(self):
        """Flush buffered traces to the Langfuse server."""
        if self.enabled and self.client:
            try:
                self.client.flush()
            except Exception as e:
                logger.error(f"Error flushing Langfuse: {e}")
