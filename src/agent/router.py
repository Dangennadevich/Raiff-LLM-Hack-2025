
from src.agent.state import State

def router(state: State) -> str:
    print('----', 'router')
    if state["sufficient"]:
        return "final_answer"

    if state["iteration"] >= state["max_iterations"]:
        return "final_answer"

    return "rag_step"