from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from src.agent.tool import judge_rag_sufficiency

load_dotenv()

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

llm_with_tools = ChatOpenAI(
    model="google/gemini-2.5-flash-lite",
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0,
    api_key=OPENROUTER_API_KEY
).bind_tools([judge_rag_sufficiency])

llm_final_answer = ChatOpenAI(
        model="qwen/qwen3-32b",
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0,
        api_key=OPENROUTER_API_KEY
    )