from langgraph.graph import StateGraph, END
from app.agents.graph.state import GraphState
from app.agents.graph.nodes import (
    retrieve_node,
    grade_node,
    generate_node,
    rewrite_node,
    decide_to_generate,
)

_builder = StateGraph(GraphState)

_builder.add_node("retrieve", retrieve_node)
_builder.add_node("grade", grade_node)
_builder.add_node("generate", generate_node)
_builder.add_node("rewrite", rewrite_node)

_builder.set_entry_point("retrieve")
_builder.add_edge("retrieve", "grade")
_builder.add_conditional_edges(
    "grade",
    decide_to_generate,
    {"generate": "generate", "rewrite": "rewrite"},
)
_builder.add_edge("rewrite", "retrieve")
_builder.add_edge("generate", END)

graph = _builder.compile()
