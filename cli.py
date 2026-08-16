"""Command-line interface for the educational chatbot."""

from bot import classify


def main() -> None:
    print("Помощник аналитика маркетинга Поиска. Для выхода введите: выход")
    while True:
        try:
            text = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nДо свидания!")
            return
        if text.casefold() in {"выход", "exit", "quit"}:
            print("Бот: До свидания!")
            return
        response = classify(text)
        print(f"Бот [{response.title}, confidence={response.confidence:.0%}]: {response.answer}")


if __name__ == "__main__":
    main()

