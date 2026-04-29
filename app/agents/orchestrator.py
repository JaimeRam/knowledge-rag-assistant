from app.rag.llm_client import LLMClient
from app.mcp.server import MCPServer
from app.agents.state import AgentState
from typing import Dict, Any, Optional
import logging
import json

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Orchestrate multi-step agent workflows with tool calling."""
    
    def __init__(self):
        self.llm_client = LLMClient()
        self.mcp_server = MCPServer()
    
    async def process_query(self, query: str, conversation_id: str) -> Dict[str, Any]:
        """Process a query with agent orchestration."""
        state = AgentState(conversation_id=conversation_id)
        state.add_message("user", query)
        
        # Step 1: Determine if tools are needed
        tool_decision = await self._decide_tool_usage(query)
        
        if tool_decision["use_tool"]:
            # Step 2: Call the appropriate tool
            tool_result = await self.mcp_server.call_tool(
                tool_decision["tool_name"],
                **tool_decision["parameters"]
            )
            
            state.add_tool_call(
                tool_decision["tool_name"],
                tool_decision["parameters"],
                tool_result
            )
            
            # Step 3: Generate response with tool context
            prompt = self._build_prompt_with_tool_context(query, tool_result)
        else:
            # Direct RAG response
            from app.rag.retriever import Retriever
            from app.rag.prompt_builder import PromptBuilder
            
            retriever = Retriever()
            context_chunks = await retriever.retrieve(query, limit=5)
            prompt = PromptBuilder.build_chat_prompt(query, context_chunks)
            await retriever.close()
        
        # Step 4: Generate final response
        llm_response = await self.llm_client.chat(prompt)
        state.add_message("assistant", llm_response["content"])
        
        return {
            "response": llm_response["content"],
            "state": state.to_dict(),
            "tool_used": tool_decision.get("tool_name") if tool_decision.get("use_tool") else None,
            "token_usage": llm_response["usage"]
        }
    
    async def _decide_tool_usage(self, query: str) -> Dict[str, Any]:
        """Decide if a tool should be used and which one."""
        # Simple heuristic-based decision (in production, use LLM for this)
        query_lower = query.lower()
        
        if "by name" in query_lower or "called" in query_lower:
            # Extract name from query
            words = query_lower.split()
            name = words[-1] if words else ""
            return {
                "use_tool": True,
                "tool_name": "get_digimon_by_name",
                "parameters": {"name": name.capitalize()}
            }
        elif "level" in query_lower or "rookie" in query_lower or "champion" in query_lower:
            # Extract level
            levels = ["fresh", "in training", "rookie", "champion", "ultimate", "mega"]
            level = next((l for l in levels if l in query_lower), "rookie")
            return {
                "use_tool": True,
                "tool_name": "get_digimon_by_level",
                "parameters": {"level": level.capitalize()}
            }
        elif "skills" in query_lower:
            # Need ID first - this would require multi-step in production
            return {"use_tool": False}
        else:
            return {"use_tool": False}
    
    def _build_prompt_with_tool_context(self, query: str, tool_result: Dict[str, Any]) -> str:
        """Build prompt with tool result context."""
        context = json.dumps(tool_result, indent=2)
        
        return f"""You are a Digimon expert assistant. The user asked: "{query}"

I used a tool to retrieve the following information:
{context}

Please provide a helpful and informative response based on this data."""
    
    async def close(self):
        """Close resources."""
        await self.llm_client.close()
        await self.mcp_server.close()
