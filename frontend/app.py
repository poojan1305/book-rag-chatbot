import streamlit as st
import httpx
import uuid

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Book Wisdom Chatbot",
    page_icon="📚",
    layout="centered"
)

st.title("📚 Book Wisdom Chatbot")
st.caption("Ask anything from The Almanack of Naval Ravikant")


if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("📖 Books Loaded")
    st.success("The Almanack of Naval Ravikant")
    st.success("The Way of the Superior Man")

    st.divider()
    st.caption("More books coming soon...")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask Naval anything..."):
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching the book..."):
            try:
                response = httpx.post(
                    f"{API_URL}/chat",
                    json={
                        "question": prompt,
                        "session_id": st.session_state.session_id
                    },
                    timeout=30
                )
                answer = response.json()["answer"]
            except Exception as e:
                answer = f"Error connecting to backend: {str(e)}"

        st.markdown(answer)
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })