from src.agent.graph import app
import logging

logging.basicConfig(
    level=logging.INFO,                 
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]  # вывод в консоль
)

logger = logging.getLogger(__name__)


# | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = | 6. Call LLM | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = | = = |

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true", help="Run interactive chat mode")
    args = parser.parse_args()

    logger.info("[__main__] Start...")

    # === ИНТЕРАКТИВНЫЙ РЕЖИМ ===
    print("⚡ Запущен интерактивный режим. Введите вопрос.")
    print("Введите `exit` чтобы выйти.\n")

    while True:
        user_input = input("> ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Выход.")
            break

        initial_state = {
            "user_question": user_input, # "Какие налоги уплачиваются с вклада?",
            "rag_data": [],
            "sufficient": False,
            "followup_query": None,
            "confidence": 0.0,
            "iteration": 0,
            "max_iterations": 3,
            "final_answer": None,
        }
        # -------------------------
        # 2. Запускаем graph
        # -------------------------
        state = app.invoke(initial_state)
        # -------------------------
        # 3. Проверяем результат
        # -------------------------
        logger.info("Финальное состояние агента:\n")
        logger.info(state)
        logger.info(f'\nAssistant:\n{state['final_answer']}\n')
