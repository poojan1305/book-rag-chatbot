import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.retriever import get_retriever

load_dotenv()

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def format_docs(docs):
    formatted = []
    for doc in docs:
        formatted.append(
            f"[{doc.metadata['book_title']} | Page {doc.metadata['page']}]\n{doc.page_content}"
        )
    return "\n\n".join(formatted)

def build_chain():
    llm = ChatAnthropic(
        model="claude-sonnet-4-6",
        temperature=0.1,
        max_tokens=1000
    )

    retriever = get_retriever()

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a wisdom assistant trained on books by great thinkers.
Answer ONLY using the provided context from the books.
Always mention which book and page number your answer comes from.
If the context does not cover the question say:
"This topic isn't covered in the books I was trained on."
Never make up information. Be concise and insightful.

Context:
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"]))
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history"
    )

    return chain_with_history

def ask(chain, question: str, session_id: str = "default"):
    response = chain.invoke(
        {"question": question},
        config={"configurable": {"session_id": session_id}}
    )
    print(f"\nQuestion: {question}")
    print(f"\nAnswer: {response}")

if __name__ == "__main__":
    print("Building chain...")
    chain = build_chain()
    ask(chain, "What does Naval say about HEALTH?")
    # ask(chain, "How can I be happier according to Naval?")