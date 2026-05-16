from typing import TypedDict, List, Optional


class GraphState(TypedDict):
    query: str
    rewritten_query: Optional[str]
    documents: List[dict]
    generation: str
    relevant: bool
    iterations: int
