import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3:1.7b"


def ask_llm(question: str) -> str:
    if not question.strip():
        raise ValueError("Question must not be empty.")

    payload = {
        "model": MODEL,
        "prompt": question,
        "stream": False,
        "think": False,
        "keep_alive": True,
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=(5, 180),
    )

    response.raise_for_status()

    data = response.json()
    total_seconds = data["total_duration"] / 1_000_000_000
    load_seconds = data["load_duration"] / 1_000_000_000
    eval_seconds = data["eval_duration"] / 1_000_000_000
    output_tokens = data["eval_count"]
    generation_speed = output_tokens / eval_seconds if eval_seconds > 0 else 0

    print(f"\nTotal time: {total_seconds:.2f} seconds")
    print(f"Model load time: {load_seconds:.2f} seconds")
    print(f"Output tokens: {output_tokens}")
    print(f"Generation speed: {generation_speed:.2f} tokens/second")

    return data["response"]


if __name__ == "__main__":
    question = input("Ask NASA AI: ")

    try:
        answer = ask_llm(question)
        print("\nANSWER:")
        print(answer)
    except requests.exceptions.Timeout:
        print("The request timed out. Please try again.")
    except requests.exceptions.ConnectionError:
        print("Could not connect to Ollama. Make sure it is running.")
    except requests.exceptions.HTTPError as error:
        print(f"Ollama returned an HTTP error: {error}")
    except ValueError as error:
        print(f"Invalid input: {error}")
    except requests.exceptions.InvalidURL as error:
        print(f"Invalid Ollama URL configuration: {error}")
