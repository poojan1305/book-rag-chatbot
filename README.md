📚 Book Wisdom Chatbot — RAG Pipeline
A conversational AI chatbot that answers questions based on the wisdom of 5 books, powered by Retrieval-Augmented Generation (RAG), LangChain, ChromaDB, and Claude (Anthropic).
---
🧠 What is RAG?
RAG (Retrieval-Augmented Generation) is a technique where an LLM doesn't answer from its own training data — it answers from your data, retrieved in real time.
Instead of fine-tuning a model (expensive), you:
Store your documents as vectors in a database
At query time, fetch the most relevant chunks
Send those chunks to the LLM as context
LLM answers only from that context
---
🔄 Complete Workflow
Phase 1 — Offline Ingestion (run once)
```
PDF Books
   ↓
PyMuPDF extracts raw text page by page
   ↓
LangChain RecursiveCharacterTextSplitter cuts text into chunks
   → chunk_size = 500 tokens (~375 words)
   → chunk_overlap = 50 tokens (so ideas don't get cut off at boundaries)
   ↓
Each chunk gets metadata attached:
   → book_title, author, page number, chunk_id
   ↓
HuggingFace all-MiniLM-L6-v2 converts each chunk into a vector
   → 384 numbers representing the meaning of that chunk
   ↓
All vectors + text + metadata stored in ChromaDB (local)
   → vectorstore/ folder on disk
```
This runs once. After ingestion, your books live in the vectorstore permanently.
---
Phase 2 — Runtime Query Flow (every user message)
```
User types: "What does Naval say about wealth?"
   ↓
Same HuggingFace model converts the question into a vector
   → [0.21, -0.85, 0.39 ... 384 numbers]
   ↓
ChromaDB does cosine similarity search
   → compares query vector against all ~2600 chunk vectors
   → returns top 5 most similar chunks (no LLM involved, just math)
   ↓
LangChain assembles the prompt:
   → System prompt (instructions for Claude)
   → 5 retrieved chunks (the context)
   → Conversation history (last 5 exchanges)
   → User question
   ↓
Full prompt (~2800 tokens) sent to Claude via Anthropic API
   ↓
Claude reads the context and generates an answer
   → Cites book title and page number
   → Never makes up information outside the context
   ↓
Answer streamed back to Streamlit UI
   ↓
Exchange saved to memory for next question
```
---
🗂️ Project Structure
```
book-rag-chatbot/
├── data/
│   └── books/              # PDF books go here
├── vectorstore/            # ChromaDB persists here (auto-created)
├── backend/
│   ├── __init__.py
│   ├── ingest.py           # One-time ingestion pipeline
│   ├── retriever.py        # Vector similarity search logic
│   ├── rag_chain.py        # LangChain chain + Claude + memory
│   └── main.py             # FastAPI REST API
├── frontend/
│   └── app.py              # Streamlit chat UI
├── .env                    # ANTHROPIC_API_KEY (never commit this)
├── .gitignore
├── requirements.txt
└── README.md
```
---
⚙️ Tech Stack
Layer	Tool	Purpose
PDF Extraction	PyMuPDF (fitz)	Extract raw text from books
Text Splitting	LangChain TextSplitter	Chunk text into 500-token pieces
Embeddings	HuggingFace all-MiniLM-L6-v2	Convert text to vectors (free, local)
Vector Database	ChromaDB	Store and search vectors locally
Orchestration	LangChain 1.x (LCEL)	Connect retrieval → prompt → LLM
LLM	Claude Sonnet 4.6 (Anthropic)	Generate answers from context
Memory	RunnableWithMessageHistory	Remember last 5 chat exchanges
Backend	FastAPI + Uvicorn	REST API serving the chain
Frontend	Streamlit	Chat UI in the browser
---
📖 Books Indexed
The Almanack of Naval Ravikant — Eric Jorgenson
The Way of the Superior Man — David Deida
Psycho Cybernetics — Maxwell Maltz
What Young India Wants — Chetan Bhagat
---
🚀 Setup & Installation
1. Clone the repo
```bash
git clone https://github.com/your-username/book-rag-chatbot.git
cd book-rag-chatbot
```
2. Create virtual environment
```bash
python -m venv venv
source venv/Scripts/activate    # Windows (Git Bash)
source venv/bin/activate        # Mac/Linux
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
4. Add your API key
Create a `.env` file in the root:
```env
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxx
```
5. Add PDFs
Drop your book PDFs into `data/books/` and make sure filenames match `BOOK_METADATA` in `ingest.py`.
6. Run ingestion (one time only)
```bash
python backend/ingest.py
```
7. Start backend
```bash
uvicorn backend.main:app --reload
```
8. Start frontend (new terminal)
```bash
source venv/Scripts/activate
streamlit run frontend/app.py
```
Open `http://localhost:8501` in your browser.
---
💡 Key Concepts
Tokens
A token is roughly 0.75 words
500 tokens ≈ 375 words per chunk
Each query costs ~2800 input tokens + ~300 output tokens
Cost per query ≈ $0.012 using Claude Sonnet 4.6
Embeddings
Each chunk is converted to 384 numbers (all-MiniLM-L6-v2)
Same model used for both ingestion and query — must match
Vectors capture semantic meaning, not just keywords
Cosine Similarity
How ChromaDB finds relevant chunks
Measures angle between two vectors
Smaller angle = more similar meaning
Temperature
Set to 0.1 for factual, grounded answers
Does not affect cost — only affects response style
Low temperature = Claude sticks to the retrieved context
---
💰 Cost Estimate
Usage	Queries	Cost
Testing & learning	~200	~$2.60
Demo to friends	~100	~$1.30
Full project	~350	~$4.55
$5 in Anthropic credits is enough for the entire project.
---
📡 API Endpoints
Method	Endpoint	Description
GET	`/`	Health check
POST	`/chat`	Send a question, get an answer
GET	`/books`	List all indexed books
Example `/chat` request
```json
{
  "question": "What does Naval say about wealth?",
  "session_id": "user-123"
}
```
Example response
```json
{
  "answer": "According to Naval (Page 29): Making money is not a thing you do — it's a skill you learn..."
}
```
---
🔮 Future Improvements
[ ] Add re-ranking with Cohere for better retrieval quality
[ ] Deploy to Render for mobile access
[ ] Add book filter in UI (search only selected books)
[ ] Add streaming responses token by token
[ ] Add more books to the knowledge base