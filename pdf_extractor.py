import pymupdf

document = pymupdf.open("data/raw/mars_report.pdf")

print("Number of pages:", len(document))

page = document[0]

text = page.get_text("text", sort=True)

print(text[:500])

for page_number, page in enumerate(document, start=0):
    text = page.get_text("text", sort=True)
    char_count = len(text)
    print(f"\nPage {page_number}: {char_count} characters")
    print(f"Preview: {text[:500]}")

document.close()
