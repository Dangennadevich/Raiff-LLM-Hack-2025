from langgraph.graph import StateGraph, END
from src.agent.nodes import first_call_rag, judge_step, rag_step, final_answer_step
from src.agent.state import State
from src.agent.router import router

graph = StateGraph(State)

graph.add_node("first_call_rag", first_call_rag)
graph.add_node("judge", judge_step)
graph.add_node("rag_step", rag_step)
graph.add_node("final_answer", final_answer_step)

graph.set_entry_point("first_call_rag")

graph.add_edge("first_call_rag", "judge")

graph.add_conditional_edges(
    "judge",
    router,
    {
        "rag_step": "rag_step",
        "final_answer": "final_answer",
    }
)

graph.add_edge("rag_step", "judge")
graph.add_edge("final_answer", END)

app = graph.compile()