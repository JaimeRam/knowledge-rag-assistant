from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.rag.retriever import Retriever
from app.rag.prompt_builder import PromptBuilder
from app.rag.llm_client import LLMClient
from app.db.redis import RedisManager
from app.observability.langfuse import LangfuseManager
import time
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["chat"])


class ChatRequest(BaseModel):
    query: str
    level_filter: Optional[str] = None
    type_filter: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    context_sources: list
    token_usage: dict
    latency_ms: int


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint with RAG."""
    start_time = time.time()
    
    retriever = Retriever()
    llm_client = LLMClient()
    redis = RedisManager()
    langfuse = LangfuseManager()
    
    try:
        # Check cache first
        cache_key = f"chat:{request.query}"
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
        await langfuse.trace_retrieval(
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
        
        # Trace chat interaction
        await langfuse.trace_chat(
            query=request.query,
            response=llm_response["content"],
            context_chunks=context_chunks,
            token_usage=llm_response["usage"],
            latency_ms=latency_ms
        )
        
        # Cache the response
        await redis.set(cache_key, response.dict(), ttl=86400)
        
        # Flush traces
        langfuse.flush()
        
        return response
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    
    finally:
        await retriever.close()
        await llm_client.close()
        langfuse.flush()
