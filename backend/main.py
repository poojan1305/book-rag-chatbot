import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.rag_chain import build_chain

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

chain = build_chain()

class QueryRequest(BaseModel):
    question: str
    session_id: str = "default"

@app.get("/")
def root():
    return {"status": "Book RAG Chatbot is running"}

@app.post("/chat")
def chat(request: QueryRequest):
    response = chain.invoke(
        {"question": request.question},
        config={"configurable": {"session_id": request.session_id}}
    )
    return {"answer": response}

@app.get("/books")
def get_books():
    return {
        "books": [
            "The Almanack of Naval Ravikant"
        ]
    }