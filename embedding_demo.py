from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="nomic-embed-text", base_url="http://localhost:11434"
)

texts = [
    "The Maaz and Seitah formations are igneous in origin.",
    "Aqueous alteration produced carbonates, iron oxides and sulfates.",
    "Perseverance collected rock cores from the Jezero crater floor.",
]

question = "Which minerals formed due to water alteration?"

# Convert document passages into vectors
document_vectors = embeddings.embed_documents(
    [f"search_document: {text}" for text in texts]
)

# Convert the question into a vector
question_vector = embeddings.embed_query(f"search_query: {question}")

print("Number of document vectors:", len(document_vectors))
print("Dimensions per vector:", len(document_vectors[0]))
print("Question vector dimensions:", len(question_vector))

# TODO:
# Calculate cosine similarity between the question
# vector and each document vector.


from math import sqrt


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Vector dimensions must match")

    # TODO 1: Calculate the dot product.
    # Hint: sum(x * y for x, y in zip(a, b))
    dot_product = sum(x * y for x, y in zip(a, b))

    # TODO 2: Calculate the magnitude of a.
    # Hint: sqrt(sum(x * x for x in a))
    magnitude_a = sqrt(sum(x * x for x in a))
    # TODO 3: Calculate the magnitude of b.
    # Hint: sqrt(sum(x * x for x in b))
    magnitude_b = sqrt(sum(x * x for x in b))

    # TODO 4: Reject zero-magnitude vectors.
    if magnitude_a == 0 or magnitude_b == 0:
        raise ValueError("Zero-magnitude vectors are not allowed")

    # TODO 5: Return the cosine similarity.
    return dot_product / (magnitude_a * magnitude_b)

    raise NotImplementedError


# TODO:
# Print the passages ranked from highest
# to lowest similarity.

results = []

for text, vector in zip(texts, document_vectors):
    score = cosine_similarity(question_vector, vector)
    results.append((score, text))

results.sort(key=lambda item: item[0], reverse=True)

for rank, (score, text) in enumerate(results, start=1):
    print(f"{rank}. Similarity: {score:.4f}")
    print(f"   {text}")
