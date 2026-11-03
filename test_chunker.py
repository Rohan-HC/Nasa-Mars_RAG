from chunker import chunk_text

# Test 1: Normal overlapping chunks
result = chunk_text(
    "Mars rovers explore ancient rocks and search for evidence of past water",
    chunk_size=5,
    overlap=2,
)

assert result == [
    "Mars rovers explore ancient rocks",
    "ancient rocks and search for",
    "search for evidence of past",
    "of past water",
]

# Test 2: No overlap
assert chunk_text("one two three four five six", chunk_size=3, overlap=0) == [
    "one two three",
    "four five six",
]

# Test 3: Empty input
assert chunk_text("", 5, 2) == []

# Test 4: Invalid parameters must fail
for size, overlap in [
    (0, 0),
    (5, 5),
    (5, -1),
    (-3, 0),
]:
    try:
        chunk_text("one two three", size, overlap)
    except ValueError:
        pass
    else:
        raise AssertionError(f"Expected ValueError for {size=}, {overlap=}")

print("All chunking tests passed!")
