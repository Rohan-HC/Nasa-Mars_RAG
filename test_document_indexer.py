import hashlib
from pathlib import Path

from document_indexer import index_uploaded_pdf

PDF_PATH = Path("data/uploads/f9ca9aece33040389fab30ef6e958eaa.pdf")

if not PDF_PATH.exists():
    raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

# Calculate SHA-256 from the uploaded PDF
hasher = hashlib.sha256()

with PDF_PATH.open("rb") as file:
    for block in iter(lambda: file.read(1024 * 1024), b""):
        hasher.update(block)

file_hash = hasher.hexdigest()

print("PDF:", PDF_PATH)
print("SHA-256:", file_hash)

print("\n--- FIRST INDEXING ATTEMPT ---")

first = index_uploaded_pdf(
    pdf_path=PDF_PATH,
    file_hash=file_hash,
    original_name="mars_report.pdf",
)

print(first)

print("\n--- SECOND INDEXING ATTEMPT ---")

second = index_uploaded_pdf(
    pdf_path=PDF_PATH,
    file_hash=file_hash,
    original_name="mars_report.pdf",
)

print(second)

assert first["status"] in (
    "indexed",
    "already_indexed",
)

assert second["status"] == "already_indexed"
assert first["chunks"] == second["chunks"]

print("\nINTEGRATION TEST PASSED")
