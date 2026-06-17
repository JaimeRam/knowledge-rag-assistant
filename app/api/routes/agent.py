from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from app.agents.pydantic_agent import agent
from app.agents.graph.workflow import graph
from app.observability.langfuse import LangfuseManager
from app.rag.llm_client import LLMClient
from app.guardrails.input_guard import InputGuard, OFF_TOPIC_MESSAGE
import time
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/agent", tags=["agent"])


class AgentRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None


class AgentResponse(BaseModel):
    response: str
    tool_calls: List[str]
    usage: dict
    latency_ms: int


async def _assert_digimon_query(query: str) -> None:
    """Raise HTTP 400 if the query is not Digimon-related."""
    llm_client = LLMClient()
    try:
        guard = InputGuard(llm_client)
        if not await guard.is_digimon_related(query):
            raise HTTPException(status_code=400, detail=OFF_TOPIC_MESSAGE)
    finally:
        await llm_client.close()


@router.post("", response_model=AgentResponse)
async def run_agent(request: AgentRequest):
    """PydanticAI agent endpoint with automatic tool selection."""
    start_time = time.time()
    langfuse = LangfuseManager()

    try:
        await _assert_digimon_query(request.query)
        result = await agent.run(request.query)
        latency_ms = int((time.time() - start_time) * 1000)

        # Extract tool names from message history
        tool_calls = []
        for msg in result.all_messages():
            for part in msg.parts:
                if hasattr(part, "tool_name"):
                    tool_calls.append(part.tool_name)

        usage = result.usage
        usage_dict = {
            "input_tokens": (usage.input_tokens if usage else 0) or 0,
            "output_tokens": (usage.output_tokens if usage else 0) or 0,
            "total_tokens": (usage.total_tokens if usage else 0) or 0,
        }

        langfuse.trace_chat(
            query=request.query,
            response=result.output,
            context_chunks=[],
            token_usage={
                "prompt_tokens": usage_dict["input_tokens"],
                "completion_tokens": usage_dict["output_tokens"],
                "total_tokens": usage_dict["total_tokens"],
            },
            latency_ms=latency_ms,
            metadata={"tool_calls": tool_calls, "agent": "pydantic_ai"},
        )
        langfuse.flush()

        return AgentResponse(
            response=result.output,
            tool_calls=tool_calls,
            usage=usage_dict,
            latency_ms=latency_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in agent endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


class GraphRequest(BaseModel):
    query: str


class GraphResponse(BaseModel):
    response: str
    iterations: int
    documents_used: int
    latency_ms: int


@router.post("/graph", response_model=GraphResponse)
async def run_graph(request: GraphRequest):
    """Corrective RAG endpoint: retrieve → grade → generate (or rewrite query and retry)."""
    start_time = time.time()
    langfuse = LangfuseManager()

    try:
        await _assert_digimon_query(request.query)
        initial_state = {
            "query": request.query,
            "rewritten_query": None,
            "documents": [],
            "generation": "",
            "relevant": False,
            "iterations": 0,
        }
        final_state = await graph.ainvoke(initial_state)
        latency_ms = int((time.time() - start_time) * 1000)

        langfuse.trace_chat(
            query=request.query,
            response=final_state["generation"],
            context_chunks=final_state["documents"],
            token_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            latency_ms=latency_ms,
            metadata={
                "agent": "langgraph_corrective_rag",
                "iterations": final_state["iterations"],
                "rewritten_query": final_state.get("rewritten_query"),
            },
        )
        langfuse.flush()

        return GraphResponse(
            response=final_state["generation"],
            iterations=final_state["iterations"],
            documents_used=len(final_state["documents"]),
            latency_ms=latency_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in graph endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
