from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from app.rag.retriever import Retriever
from app.rag.prompt_builder import PromptBuilder
from app.rag.llm_client import LLMClient
from app.db.redis import RedisManager
from app.observability.langfuse import LangfuseManager
from app.observability.llm_judge import LLMJudge
from app.guardrails.input_guard import InputGuard, OFF_TOPIC_MESSAGE
import hashlib
import json
import time
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["chat"])


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    level_filter: Optional[str] = None
    type_filter: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    context_sources: list
    token_usage: dict
    latency_ms: int


async def _judge_response(trace_id: str, query: str, context_texts: list, answer: str) -> None:
    """Background task: LLM-as-judge scores sent to Langfuse after the response is returned."""
    judge = LLMJudge()
    langfuse = LangfuseManager()
    try:
        logger.info(f"[judge] evaluating trace {trace_id[:8]}…")
        scores = await judge.evaluate(query, context_texts, answer)
        if scores:
            for name, value in scores.items():
                langfuse.add_score(trace_id, name, value, comment="llm-judge")
            logger.info(f"[judge] scores added to trace {trace_id[:8]}: {scores}")
        else:
            logger.warning(f"[judge] evaluate() returned empty dict for trace {trace_id[:8]}")
    except Exception as e:
        logger.error(f"[judge] background task failed: {e}")
    finally:
        langfuse.flush()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """Chat endpoint with RAG."""
    start_time = time.time()
    
    retriever = Retriever()
    llm_client = LLMClient()
    redis = RedisManager()
    langfuse = LangfuseManager()
    
    try:
        # Guardrail: block off-topic queries before any expensive operations
        guard = InputGuard(llm_client)
        if not await guard.is_digimon_related(request.query):
            raise HTTPException(status_code=400, detail=OFF_TOPIC_MESSAGE)

        # Check cache first — include filters in the key to avoid collisions
        query_hash = hashlib.sha256(request.query.encode()).hexdigest()[:16]
        filters_hash = hashlib.md5(
            json.dumps({"level": request.level_filter, "type": request.type_filter}, sort_keys=True).encode()
        ).hexdigest()[:8]
        cache_key = f"chat:{query_hash}:{filters_hash}"
        cached_response = await redis.get(cache_key)
        
        if cached_response:
            logger.info(f"Cache hit for query: {request.query}")
            return ChatResponse(**cached_response)
        
        # Build filters if provided
        filters = None
        if request.level_filter or request.type_filter:
            filters = {}
            if request.level_filter:
                filters["level"] = request.level_filter
            if request.type_filter:
                filters["type"] = request.type_filter
        
        # Retrieve relevant chunks
        retrieval_start = time.time()
        context_chunks = await retriever.retrieve(request.query, limit=5, filters=filters)
        retrieval_latency = int((time.time() - retrieval_start) * 1000)
        
        # Trace retrieval
        avg_score = sum(c.get("score", 0) for c in context_chunks) / len(context_chunks) if context_chunks else 0
        langfuse.trace_retrieval(
            query=request.query,
            retrieved_count=len(context_chunks),
            avg_score=avg_score,
            latency_ms=retrieval_latency
        )
        
        # Build prompt
        prompt = PromptBuilder.build_chat_prompt(request.query, context_chunks)
        
        # Get LLM response
        llm_response = await llm_client.chat(prompt)
        
        # Calculate latency
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Extract context sources
        context_sources = [
            {
                "digimon_id": chunk.get("payload", {}).get("digimon_id"),
                "name": chunk.get("payload", {}).get("name"),
                "score": chunk.get("score")
            }
            for chunk in context_chunks
        ]
        
        # Build response
        response = ChatResponse(
            response=llm_response["content"],
            context_sources=context_sources,
            token_usage=llm_response["usage"],
            latency_ms=latency_ms
        )
        
        # Trace chat interaction and schedule LLM-as-judge in background
        trace_id = langfuse.trace_chat(
            query=request.query,
            response=llm_response["content"],
            context_chunks=context_chunks,
            token_usage=llm_response["usage"],
            latency_ms=latency_ms,
        )
        if trace_id:
            context_texts = [
                c.get("payload", {}).get("chunk_text", "")
                for c in context_chunks
                if c.get("payload", {}).get("chunk_text")
            ]
            if context_texts:
                background_tasks.add_task(
                    _judge_response, trace_id, request.query, context_texts, llm_response["content"]
                )

        # Cache the response
        await redis.set(cache_key, response.model_dump(), ttl=86400)

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

    finally:
        await retriever.close()
        await llm_client.close()
        await redis.close()
        langfuse.flush()
