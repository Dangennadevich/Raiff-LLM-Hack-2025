from src.agent.schemas import ChatResponse, ChatRequest
from src.agent.graph import app as agent_app
from fastapi import FastAPI, HTTPException
from uuid import uuid4

from src.infra.logging import setup_logging
import logging

setup_logging()

logger = logging.getLogger("rag-agent")
logger.info("Application starting...")

# FastAPI
api = FastAPI(
    title="Agentic RAG Service",
    version="0.1.1",
)

# Routes
@api.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid4())

    logger.info(
        "Incoming request",
        extra={
            "session_id": session_id,
            "user_question": req.message,
        },
    )

    initial_state = {
        "session_id" : session_id,
        "user_question": req.message,
        "rag_data": [],
        "sufficient": False,
        "followup_query": None,
        "confidence": 0.0,
        "iteration": 0,
        "max_iterations": 3,
        "final_answer": None,
    }

    try:
        state = agent_app.invoke(initial_state)
    except Exception as e:
        logger.exception("Agent execution failed")
        raise HTTPException(status_code=500, detail="Agent execution failed")

    logger.info(
        "Agent finished",
        extra={
            "session_id": session_id,
            "user_question": state.get("user_question"),
            "rag_data": state.get("rag_data"),
            "final_answer": state.get("final_answer"),
            "iterations": state.get("iteration"),
            "confidence": state.get("confidence"),
        },
    )

    return ChatResponse(
        session_id=session_id,
        answer=state.get("final_answer"),
        confidence=state.get("confidence", 0.0),
        iterations=state.get("iteration", 0),
    )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:api",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )