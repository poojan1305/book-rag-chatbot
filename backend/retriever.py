import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

VECTORSTORE_DIR = "vectorstore"

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=embeddings
    )
    return vectorstore

def get_retriever(filter_book: str = None):
    vectorstore = load_vectorstore()
    
    search_kwargs = {"k": 5}  # top 5 chunks per query
    
    if filter_book:
        search_kwargs["filter"] = {"book_title": filter_book}
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs
    )
    return retriever

def test_retriever(query: str):
    retriever = get_retriever()
    docs = retriever.invoke(query)
    
    print(f"\nQuery: {query}")
    print(f"Found {len(docs)} chunks\n")
    
    for i, doc in enumerate(docs):
        print(f"--- Chunk {i+1} ---")
        print(f"Book: {doc.metadata['book_title']}")
        print(f"Page: {doc.metadata['page']}")
        print(f"Text: {doc.page_content[:200]}...")
        print()

if __name__ == "__main__":
    test_retriever("What does Naval say about HEALTH?")