import os
import fitz  # pymupdf
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

load_dotenv()

BOOKS_DIR = "data/books"
VECTORSTORE_DIR = "vectorstore"

BOOK_METADATA = {
    "naval.pdf": {
        "title": "The Almanack of Naval Ravikant",
        "author": "Eric Jorgenson"
    },
}

def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        # skip first 25 and last 10 pages
        if page_num < 25 or page_num > (total_pages - 10):
            continue
        if text.strip():
            pages.append({
                "text": text,
                "page": page_num + 1
            })
    doc.close()
    return pages

def ingest_books():
    print("Starting ingestion...\n")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )

    all_documents = []

    for filename in os.listdir(BOOKS_DIR):
        if not filename.endswith(".pdf"):
            continue

        pdf_path = os.path.join(BOOKS_DIR, filename)
        meta = BOOK_METADATA.get(filename, {"title": filename, "author": "Unknown"})

        print(f"Processing: {meta['title']}")
        pages = extract_text_from_pdf(pdf_path)
        print(f"  Pages extracted: {len(pages)}")

        for page_data in pages:
            chunks = splitter.split_text(page_data["text"])
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "book_title": meta["title"],
                        "author": meta["author"],
                        "page": page_data["page"],
                        "source": filename,
                        "chunk_id": f"{filename}_p{page_data['page']}_c{i}"
                    }
                )
                all_documents.append(doc)

        print(f"  Chunks created: {sum(1 for d in all_documents if d.metadata['source'] == filename)}\n")

    print(f"Total chunks to embed: {len(all_documents)}")
    print("Embedding and storing in Chroma... (this may take a minute)")

    vectorstore = Chroma.from_documents(
        documents=all_documents,
        embedding=embeddings,
        persist_directory=VECTORSTORE_DIR
    )

    print(f"\nDone! {len(all_documents)} chunks stored in '{VECTORSTORE_DIR}'")
    return vectorstore

if __name__ == "__main__":
    ingest_books()