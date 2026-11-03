import pymupdf

with pymupdf.open("data/raw/mars_report.pdf") as document:
    page = document[1]  # Page 2

    # Method 1: Original PDF text order
    original = page.get_text("text", sort=False)

    # Method 2: Position-sorted text
    sorted_text = page.get_text("text", sort=True)

    # Method 3: Extract left and right halves separately
    midpoint = (page.rect.x0 + page.rect.x1) / 2

    left_area = pymupdf.Rect(page.rect.x0, page.rect.y0, midpoint, page.rect.y1)

    right_area = pymupdf.Rect(midpoint, page.rect.y0, page.rect.x1, page.rect.y1)

    left_text = page.get_text("text", clip=left_area, sort=True)
    right_text = page.get_text("text", clip=right_area, sort=True)

    columns_text = left_text + "\n" + right_text

    for name, content in [
        ("Original", original),
        ("Sorted", sorted_text),
        ("Column-based", columns_text),
    ]:
        print(f"\n--- {name} ---")
        print(content[:800])
