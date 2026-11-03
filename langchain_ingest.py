from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Step 1: Load the PDF
loader = PyMuPDFLoader("data/raw/mars_report.pdf", mode="page")

documents = loader.load()

print("Pages loaded:", len(documents))

# Step 2: Create a text splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)

# Step 3: Split the document
chunks = splitter.split_documents(documents)

print("Chunks created:", len(chunks))

# Step 4: Inspect the results
for chunk in chunks[:3]:
    print("\nText:")
    print(chunk.page_content[:300])

    print("\nMetadata:")
    print(chunk.metadata)
