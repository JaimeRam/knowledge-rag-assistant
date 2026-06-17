from app.mcp.tools import MCPTools
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class MCPServer:
    """MCP Server for Digimon tools."""
    
    def __init__(self):
        self.tools = MCPTools()
    
    async def call_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Call a specific MCP tool."""
        try:
            if tool_name == "get_digimon_by_name":
                name = kwargs.get("name")
                if not name:
                    return {"error": "Missing required parameter: name"}
                return await self.tools.get_digimon_by_name(name)
            
            elif tool_name == "get_digimon_by_level":
                level = kwargs.get("level")
                if not level:
                    return {"error": "Missing required parameter: level"}
                return {"results": await self.tools.get_digimon_by_level(level)}
            
            elif tool_name == "get_digimon_by_id":
                digimon_id = kwargs.get("digimon_id")
                if digimon_id is None:
                    return {"error": "Missing required parameter: digimon_id"}
                return await self.tools.get_digimon_by_id(digimon_id)

            elif tool_name == "get_digimon_skills":
                digimon_id = kwargs.get("digimon_id")
                if digimon_id is None:
                    return {"error": "Missing required parameter: digimon_id"}
                return {"skills": await self.tools.get_digimon_skills(digimon_id)}
            
            else:
                return {"error": f"Unknown tool: {tool_name}"}
        
        except Exception as e:
            logger.error(f"Error calling MCP tool {tool_name}: {e}")
            return {"error": str(e)}
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List available MCP tools."""
        return [
            {
                "name": "get_digimon_by_name",
                "description": "Get Digimon information by name",
                "parameters": {"name": "string (required)"}
            },
            {
                "name": "get_digimon_by_level",
                "description": "Get Digimons by level (e.g., Rookie, Champion)",
                "parameters": {"level": "string (required)"}
            },
            {
                "name": "get_digimon_by_id",
                "description": "Get Digimon by DAPI ID",
                "parameters": {"digimon_id": "integer (required)"}
            },
            {
                "name": "get_digimon_skills",
                "description": "Get skills for a specific Digimon",
                "parameters": {"digimon_id": "integer (required)"}
            }
        ]
    
    async def close(self):
        """Close resources."""
        await self.tools.close()
