import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3:1.7b"


def ask_llm(question: str) -> str:
    payload = {
        "model": MODEL,
        "prompt": question,
        "stream": False,
        "think": False,
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=180,
    )

    response.raise_for_status()

    data = response.json()

    return data["response"]


if __name__ == "__main__":
    question = "What was the objective of NASA's Perseverance rover?"

    answer = ask_llm(question)

    print("\nQUESTION:")
    print(question)

    print("\nMODEL ANSWER:")
    print(answer)
