"""RAG pipeline: PDF load -> chunk -> embed -> ChromaDB -> retrieve -> Groq LLM."""
import hashlib
import re  # 🔴 CHANGED: suggested questions parse karne ke liye
from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ---------- 1. Embeddings ----------
class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)  # 384-dim vectors

    def generate_embeddings(self, texts: list[str]):
        """Always returns shape (len(texts), 384)."""
        return self.model.encode(
            texts, show_progress_bar=False, normalize_embeddings=True
        )


# ---------- 2. Vector store ----------
class VectorStore:
    def __init__(self, collection_name: str = "rag_docs", persist_dir: str = "chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # distance = 1 - cosine similarity
        )

    def add_documents(self, chunks: list[str], embeddings, metadatas: list[dict]):
        # Deterministic ids: re-uploading the same PDF won't create duplicates
        ids = [
            hashlib.md5(f"{m['source']}|{m['page']}|{i}|{c}".encode()).hexdigest()
            for i, (c, m) in enumerate(zip(chunks, metadatas))
        ]
        self.collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )

    def count(self) -> int:
        return self.collection.count()

    # 🔴 CHANGED: PDF selector ke liye saare PDF names
    def list_sources(self) -> list[str]:
        data = self.collection.get(include=["metadatas"])
        return sorted({m["source"] for m in data["metadatas"] if m and "source" in m})

    # 🔴 CHANGED: suggested questions ke liye PDF ke kuch chunks (poori PDF me se barabar doori par)
    def get_chunks(self, source: str | None = None, n: int = 6) -> list[str]:
        kwargs = {"include": ["documents"]}
        if source:
            kwargs["where"] = {"source": source}
        docs = self.collection.get(**kwargs)["documents"]
        if len(docs) <= n:
            return docs
        step = len(docs) / n
        return [docs[int(i * step)] for i in range(n)]

    def reset(self):
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )


# ---------- 3. Loading + chunking ----------
def load_pdf(file, name: str | None = None) -> list[dict]:
    """`file` can be a path or a file-like object (Streamlit upload)."""
    reader = PdfReader(file)
    name = name or Path(str(file)).name
    pages = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"text": text, "source": name, "page": page_no})
    return pages


def chunk_pages(pages: list[dict], chunk_size: int = 800, chunk_overlap: int = 150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks, metadatas = [], []
    for p in pages:
        for piece in splitter.split_text(p["text"]):
            chunks.append(piece)
            metadatas.append({"source": p["source"], "page": p["page"]})
    return chunks, metadatas


def ingest_pdf(file, embedding_manager: EmbeddingManager, vector_store: VectorStore,
               name: str | None = None) -> int:
    """Full ingestion of one PDF. Returns number of chunks added."""
    pages = load_pdf(file, name)
    chunks, metadatas = chunk_pages(pages)
    if not chunks:
        return 0
    embeddings = embedding_manager.generate_embeddings(chunks)
    vector_store.add_documents(chunks, embeddings, metadatas)
    return len(chunks)


# ---------- 4. Retriever ----------
class RAGRetriever:
    def __init__(self, embedding_manager: EmbeddingManager, vector_store: VectorStore):
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store

    # 🔴 CHANGED: `source` parameter add hua, sirf ek PDF me search karne ke liye
    def retrieve(self, query: str, top_k: int = 3, score_threshold: float = 0.0,
                 source: str | None = None):
        if self.vector_store.count() == 0:
            return []

        # (1, 384) array -> [[...384 floats...]]  (exactly what Chroma expects)
        query_embedding = self.embedding_manager.generate_embeddings([query])

        extra = {"where": {"source": source}} if source else {}  # 🔴 CHANGED
        results = self.vector_store.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=min(top_k, self.vector_store.count()),
            **extra,
        )

        docs = []
        for i, (doc_id, text, meta, dist) in enumerate(zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            score = 1 - dist  # cosine similarity
            if score >= score_threshold:
                docs.append({
                    "id": doc_id,
                    "text": text,
                    "metadata": meta,
                    "similarity_score": score,
                    "rank": i + 1,
                })
        return docs


# ---------- 5. Generation ----------
# 🔴 CHANGED: prompt me chat history aur language instruction add hui
PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question using ONLY the context below.
Use the chat history only to understand follow-up questions.
If the answer is not in the context, say "I could not find this in the provided documents."
Reply in the same language as the question.

Chat history:
{history}

Context:
{context}

Question: {question}

Answer:"""


# 🔴 CHANGED: follow-up memory, pichli baatein prompt me jaati hain
def format_history(history: list[dict] | None, max_chars: int = 500) -> str:
    if not history:
        return "(none)"
    lines = []
    for m in history:
        who = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{who}: {m['content'][:max_chars]}")
    return "\n".join(lines)


# 🔴 CHANGED: "isko simple batao" jaise follow-up ko poora sawaal bana deta hai, taaki search sahi ho
def rewrite_query(query: str, history: list[dict] | None, llm) -> str:
    if not history:
        return query
    prompt = (
        "Given the chat history and a follow-up question, rewrite the follow-up as a "
        "standalone question that can be understood without the history. "
        "If it is already standalone, return it unchanged. "
        "Return ONLY the question, nothing else.\n\n"
        f"Chat history:\n{format_history(history)}\n\n"
        f"Follow-up question: {query}\n\nStandalone question:"
    )
    try:
        rewritten = llm.invoke(prompt).content.strip()
        return rewritten or query
    except Exception:
        return query


def build_prompt(query: str, docs: list[dict], history: list[dict] | None = None) -> str:
    context = "\n\n".join(
        f"[{d['metadata']['source']} - page {d['metadata']['page']}]\n{d['text']}"
        for d in docs
    )
    return PROMPT_TEMPLATE.format(
        history=format_history(history), context=context, question=query
    )


def generate_output(query: str, retriever: RAGRetriever, llm, top_k: int = 3,
                    source: str | None = None, history: list[dict] | None = None):
    """Non-streaming version. Returns (answer, retrieved_docs)."""
    docs = retriever.retrieve(rewrite_query(query, history, llm), top_k=top_k, source=source)
    if not docs:
        return "No documents found. Please upload a PDF first.", []
    return llm.invoke(build_prompt(query, docs, history)).content, docs


# 🔴 CHANGED: streaming, jawab word-by-word aata hai. Returns (docs, token_generator)
def stream_output(query: str, retriever: RAGRetriever, llm, top_k: int = 3,
                  source: str | None = None, history: list[dict] | None = None):
    search_query = rewrite_query(query, history, llm)
    docs = retriever.retrieve(search_query, top_k=top_k, source=source)
    if not docs:
        return [], iter(["No documents found. Please upload a PDF first."])

    prompt = build_prompt(query, docs, history)

    def token_generator():
        for chunk in llm.stream(prompt):
            if chunk.content:
                yield chunk.content

    return docs, token_generator()


# 🔴 CHANGED: PDF ke chunks se 3 clickable sawaal banata hai
def suggest_questions(chunks: list[str], llm, n: int = 3) -> list[str]:
    if not chunks:
        return []
    context = "\n\n".join(c[:500] for c in chunks)
    prompt = (
        f"Based on the document excerpts below, write {n} short, useful questions "
        f"a reader might ask about this document. Return exactly {n} lines, one question "
        "per line, with no numbering and no extra text.\n\n"
        f"Excerpts:\n{context}"
    )
    try:
        text = llm.invoke(prompt).content
    except Exception:
        return []
    questions = [re.sub(r"^[\s\-\*\d\.\)]+", "", line).strip() for line in text.splitlines()]
    return [q for q in questions if len(q) > 5][:n]