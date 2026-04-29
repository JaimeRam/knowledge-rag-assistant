from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import json


class AgentState(BaseModel):
    """State for agent conversations."""
    conversation_id: str
    messages: List[Dict[str, str]] = []
    current_step: str = "initial"
    context: Dict[str, Any] = {}
    tool_calls: List[Dict[str, Any]] = []
    
    def add_message(self, role: str, content: str):
        """Add a message to the conversation."""
        self.messages.append({"role": role, "content": content})
    
    def add_tool_call(self, tool_name: str, parameters: Dict[str, Any], result: Any):
        """Record a tool call."""
        self.tool_calls.append({
            "tool": tool_name,
            "parameters": parameters,
            "result": result
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return self.dict()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentState":
        """Create from dictionary."""
        return cls(**data)
