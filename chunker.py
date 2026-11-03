def chunk_text(text: str, chunk_size: int = 100, overlap: int = 20) -> list[str]:

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    words = text.split()

    step = chunk_size - overlap
    chunks = []

    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        chunks.append(" ".join(chunk_words))

        if start + chunk_size >= len(words):
            break

    return chunks


if __name__ == "__main__":
    sample = "Mars rovers explore ancient rocks and search for evidence of past water"

    chunks = chunk_text(sample, chunk_size=5, overlap=2)

    for index, chunk in enumerate(chunks, start=1):
        print(f"Chunk {index}: {chunk}")
