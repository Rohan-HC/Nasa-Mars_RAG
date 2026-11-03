from pathlib import Path

import pymupdf
import pytest


@pytest.fixture
def pdf_file(tmp_path: Path) -> Path:
    path = tmp_path / "mars-fixture.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        "Perseverance observed layered rocks in Jezero crater.",
    )
    document.save(path)
    document.close()
    return path
