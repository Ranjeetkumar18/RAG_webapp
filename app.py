import json
import os
import uuid
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from rag_pipeline import (
    EmbeddingManager,
    RAGRetriever,
    VectorStore,
    ingest_pdf,
    stream_output,        # 🔴 CHANGED: generate_output ki jagah streaming version
    suggest_questions,    # 🔴 CHANGED: suggested questions
)

load_dotenv()
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 RAG Chatbot")
st.caption("Uploads your pdfs and Ask Anything about content.")

HISTORY_FILE = "chat_history.json"
MAX_HISTORY = 3
HISTORY_WINDOW = 6  # 🔴 CHANGED: follow-up ke liye pichle itne messages (3 sawaal-jawab) yaad rakhe jaate hain


# ---------- Chat history (har chat file me save hoti hai) ----------
def load_history() -> list[dict]:
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    # purane format (sirf Q&A) wali entries ignore karo
    return [c for c in data if isinstance(c, dict) and "messages" in c]


def save_history(history: list[dict]) -> None:
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def save_current_chat() -> None:
    """Current chat ko history me daalo/update karo (sabse nayi upar aayegi)."""
    msgs = st.session_state.messages
    if not msgs:
        return
    first_q = next((m["content"] for m in msgs if m["role"] == "user"), "Chat")
    history = [c for c in load_history() if c["id"] != st.session_state.chat_id]
    history.append({
        "id": st.session_state.chat_id,
        "title": first_q[:40],
        "time": datetime.now().strftime("%d %b, %H:%M"),
        "messages": msgs,
    })
    save_history(history[-MAX_HISTORY:])  # sirf last 3 chats rakho


def new_chat() -> None:
    st.session_state.messages = []
    st.session_state.chat_id = uuid.uuid4().hex


def open_chat(chat: dict) -> None:
    st.session_state.messages = chat["messages"]
    st.session_state.chat_id = chat["id"]


# 🔴 CHANGED: suggested question par click karne se wo sawaal ban kar chala jaata hai
def ask_suggestion(question: str) -> None:
    st.session_state.pending_query = question


# 🔴 CHANGED: chat download ke liye text banata hai
def chat_to_text(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        who = "You" if m["role"] == "user" else "Bot"
        lines.append(f"{who}: {m['content']}")
        if m.get("sources"):
            lines.append("Sources: " + ", ".join(m["sources"]))
        lines.append("")
    return "\n".join(lines)


if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_id" not in st.session_state:
    st.session_state.chat_id = uuid.uuid4().hex
if "suggestions" not in st.session_state:  # 🔴 CHANGED
    st.session_state.suggestions = None


def get_api_key() -> str | None:
    key = os.getenv("GROQ_API_KEY")
    if key:
        return key
    try:  # Streamlit Cloud secrets
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return None


@st.cache_resource(show_spinner="Models loading....")
def load_components():
    embedding_manager = EmbeddingManager()
    vector_store = VectorStore()
    retriever = RAGRetriever(embedding_manager, vector_store)
    return embedding_manager, vector_store, retriever


api_key = get_api_key()
if not api_key:
    st.error("GROQ_API_KEY nahi mili. `.env` file ya Streamlit secrets me daalo.")
    st.stop()

embedding_manager, vector_store, retriever = load_components()
llm = ChatGroq(
    api_key=api_key,
    model="openai/gpt-oss-20b",
    temperature=0.2,
    max_tokens=4096,
    )

# ---------- Sidebar: documents ----------
with st.sidebar:
    st.header("📄 Documents")
    uploaded = st.file_uploader("Upload Documents", type="pdf", accept_multiple_files=True)
    if st.button("Process PDFs", disabled=not uploaded):
        for f in uploaded:
            with st.spinner(f"{f.name} processing..."):
                n = ingest_pdf(f, embedding_manager, vector_store, name=f.name)
            st.success(f"{f.name}: {n} chunks added")
        # 🔴 CHANGED: process ke baad last PDF select ho jaati hai aur uske suggested questions ban jaate hain
        last_pdf = uploaded[-1].name
        st.session_state.source_choice = last_pdf
        with st.spinner("Suggested questions loading..."):
            items = suggest_questions(vector_store.get_chunks(last_pdf), llm)
        st.session_state.suggestions = {"scope": last_pdf, "items": items}

    st.write(f"Total chunks stored: **{vector_store.count()}**")


    # 🔴 CHANGED: PDF selector, dropdown se chuno kis PDF me search karna hai
    options = ["All Docs"] + vector_store.list_sources()
    if st.session_state.get("source_choice") not in options:
        st.session_state.source_choice = "All Docs"
    source_choice = st.selectbox("🔎 Search in", options, key="source_choice")
    source = None if source_choice == "All Docs" else source_choice

    top_k = st.slider("Top K chunks", 1, 10, 4)
    if st.button("🗑️ Database clear karo"):
        vector_store.reset()
        st.session_state.suggestions = None  # 🔴 CHANGED
        new_chat()
        st.rerun()

# ---------- Main chat ----------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            st.caption("Sources: " + ", ".join(m["sources"]))

query = st.chat_input("Ask your question...")
# 🔴 CHANGED: suggested question par click hua ho to wahi sawaal use hota hai
if not query and st.session_state.get("pending_query"):
    query = st.session_state.pop("pending_query")

# 🔴 CHANGED: suggested questions, sirf khali chat par dikhte hain
if not query and not st.session_state.messages and vector_store.count() > 0:
    sug = st.session_state.suggestions
    items = sug["items"] if sug and sug["scope"] == source_choice else []
    if items:
        st.markdown("##### 💡 Suggested questions")
        for i, q in enumerate(items):
            st.button(q, key=f"sug_{i}", on_click=ask_suggestion, args=(q,),
                      use_container_width=True)
    elif st.button("💡 Sawaal suggest karo"):
        with st.spinner("Suggested questions loading..."):
            qs = suggest_questions(vector_store.get_chunks(source),llm)
        st.session_state.suggestions = {"scope": source_choice, "items": qs}
        st.rerun()

if query:
    # 🔴 CHANGED: follow-up memory, pichle messages (naya sawaal judne se pehle) bheje jaate hain
    history = list(st.session_state.messages[-HISTORY_WINDOW:])

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Documents searching..."):
            docs, token_stream = stream_output(
                query, retriever, llm, top_k, source=source, history=history
            )
        answer = st.write_stream(token_stream)  # 🔴 CHANGED: jawab word-by-word aata hai
        if not isinstance(answer, str):
            answer = "".join(str(a) for a in answer)

        sources = sorted({f"{d['metadata']['source']} (p.{d['metadata']['page']})" for d in docs})
        if docs:
            with st.expander("📚 Sources"):
                for d in docs:
                    meta = d["metadata"]
                    st.markdown(
                        f"**{meta['source']}** (page {meta['page']}) "
                        f"— score: `{d['similarity_score']:.2f}`"
                    )
                    st.caption(d["text"][:300] + "...")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
    save_current_chat()  # current chat turant history me add

# ---------- Sidebar: New chat + Recents (end me, taaki turant update ho) ----------
with st.sidebar:
    st.divider()
    st.button("➕ New Chat", on_click=new_chat, use_container_width=True, type="primary")

    # 🔴 CHANGED: chat download, poori conversation .txt me
    st.download_button(
        "⬇️ Chat download",
        data=chat_to_text(st.session_state.messages),
        file_name=f"chat_{datetime.now():%Y%m%d_%H%M}.txt",
        mime="text/plain",
        use_container_width=True,
        disabled=not st.session_state.messages,
    )

    st.markdown("""
    <style>
    div[class*="st-key-open_"] button {
        background: transparent;
        border: none;
        box-shadow: none;
        justify-content: flex-start;
        text-align: left;
        padding: 0.45rem 0.6rem;
    }
    div[class*="st-key-open_"] button p {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    div[class*="st-key-open_"] button:hover {
        background: rgba(128, 128, 128, 0.15);
    }
    /* current chat ka halka highlight */
    div[class*="st-key-open_cur_"] button {
        background: rgba(128, 128, 128, 0.22);
    }
    </style>
    """, unsafe_allow_html=True)

    st.caption("Recents")

    history_list = load_history()
    if not history_list:
        st.caption("Abhi koi history nahi hai.")
    for item in reversed(history_list):  # sabse nayi upar
        is_current = item["id"] == st.session_state.chat_id
        title = item["title"] if len(item["title"]) <= 28 else item["title"][:28] + "…"
        st.button(
            title,
            key=f"open_cur_{item['id']}" if is_current else f"open_{item['id']}",
            icon=":material/chat_bubble_outline:",
            help=item["time"],
            on_click=open_chat,
            args=(item,),
            use_container_width=True,
        )
    if history_list and st.button("History clear karo"):
        save_history([])
        new_chat()
        st.rerun()