from typing import TypedDict, List, Optional

class RagChunk(TypedDict):
    text: str
    source: Optional[str]
    query: Optional[str]
    score: Optional[float]

class State(TypedDict):
    user_question: str
    rag_data: List[str] # RagChunk

    # decision results
    sufficient: bool
    followup_query: Optional[str]
    confidence: float

    # control
    iteration: int
    max_iterations: int

    # final output
    final_answer: Optional[str]