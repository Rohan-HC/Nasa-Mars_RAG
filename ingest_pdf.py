from pathlib import Path

import pymupdf

from chunker import chunk_text


PDF_PATH = Path("data/raw/mars_report.pdf")


def build_chunks(
    pdf_path: Path, chunk_size: int = 100, overlap: int = 20
) -> list[dict]:

    records = []

    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text", sort=True)

            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

            for chunk_number, chunk in enumerate(chunks, start=1):
                record = {
                    "chunk_id": (
                        f"{pdf_path.stem}-page-{page_number}-chunk-{chunk_number}"
                    ),
                    "document": pdf_path.name,
                    "page": page_number,
                    "text": chunk,
                }

                records.append(record)

    return records


if __name__ == "__main__":
    records = build_chunks(PDF_PATH)

    print(f"Total chunks: {len(records)}")

    for record in records[:3]:
        print(record)
