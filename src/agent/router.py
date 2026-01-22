
from src.agent.logging import get_node_logger
from src.agent.state import State

import time

def router(state: State) -> str:
    logger = get_node_logger("router")

    if state["sufficient"]:
        return_values = "final_answer"
    elif state["iteration"] >= state["max_iterations"]:
        return_values = "final_answer"
    else:
        return_values = "rag_step"
    
    logger.info(
        "LLM tool call result",
        extra={
            "session_id": state.get("session_id"),
            "return_values": return_values,
        },
    )

    return return_values