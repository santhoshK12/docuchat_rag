import os
from openai import OpenAI
import io
import numpy as np
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

# ---- UI ----
st.set_page_config(page_title="DocuChat (Simple)", page_icon="📚", layout="wide")
st.title("📚 DocuChat — Simple PDF Q&A")

with st.sidebar:
    st.header("Upload PDFs")
    files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)
    ingest = st.button("🔎 Ingest / Rebuild Index")
    show_debug = st.checkbox("Show retrieved passages (debug)", value=False)

# ---- API key / client ----
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    st.error("OPENAI_API_KEY not set. In PowerShell run:\n\n$env:OPENAI_API_KEY='sk-...'\n\nThen restart the app.")
    st.stop()

client = OpenAI(api_key=API_KEY)

# ---- Globals / cache ----
@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

EMB = get_embedder()
DIM = EMB.get_sentence_embedding_dimension()

INDEX_DIR = "index"
os.makedirs(INDEX_DIR, exist_ok=True)
INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
META_PATH = os.path.join(INDEX_DIR, "meta.npy")

def extract_text_from_pdf(file_bytes: bytes) -> str:
    pdf = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for p in pdf.pages:
        text += (p.extract_text() or "")
    return text

def chunk_text(text: str, chunk_size=750, overlap=100):
    words = text.split()
    i, chunks = 0, []
    while i < len(words):
        j = min(i + chunk_size, len(words))
        chunks.append(" ".join(words[i:j]))
        if j == len(words): break
        i = j - overlap
        if i < 0: i = 0
    return chunks

def build_index(docs):
    """docs = [{'text': str, 'source': str}]"""
    texts = [d["text"] for d in docs]
    if not texts:
        return None, []
    X = EMB.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    index = faiss.IndexFlatIP(DIM)
    faiss.normalize_L2(X)
    index.add(np.array(X, dtype="float32"))
    # store (source, text)
    np.save(META_PATH, np.array([(d["source"], t) for d, t in zip(docs, texts)], dtype=object))
    faiss.write_index(index, INDEX_PATH)
    return index, texts

def load_index():
    if not (os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)):
        return None, []
    index = faiss.read_index(INDEX_PATH)
    meta = np.load(META_PATH, allow_pickle=True)
    texts = [row[1] for row in meta]
    return index, texts

def search(query: str, k=4):
    index, texts = load_index()
    if index is None:
        return []
    q = EMB.encode([query], normalize_embeddings=True)
    D, I = index.search(np.array(q, dtype="float32"), k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1: continue
        results.append({"text": texts[int(idx)], "score": float(score)})
    return results

# ---- Ingest PDFs ----
if files and ingest:
    st.info("Indexing… this may take a minute on first run.")
    docs = []
    for f in files:
        text = extract_text_from_pdf(f.read())
        for ch in chunk_text(text):
            docs.append({"text": ch, "source": f.name})
    index, _ = build_index(docs)
    if index:
        st.success(f"Indexed {len(docs)} chunks from {len(files)} file(s).")

# ---- Ask ----
st.subheader("Ask a question")
query = st.text_input("Type your question")
top_k = st.slider("Top-K passages", 1, 10, 4)

if st.button("Ask"):
    hits = search(query, k=top_k)
    if not hits:
        st.warning("No index yet or no results. Upload PDFs and click Ingest first.")
    else:
        # Optional: show retrieved passages only if debug is enabled
        if show_debug:
            st.markdown("### Top Passages (debug)")
            for i, h in enumerate(hits, start=1):
                st.markdown(f"**[{i}] score={h['score']:.3f}**\n\n> {h['text'][:600]}{'…' if len(h['text'])>600 else ''}")

        # Build context and get final answer
        ctx = "\n\n".join([f"[{i+1}] {h['text']}" for i, h in enumerate(hits)])
        prompt = f"""Answer using ONLY the context below. If not present, say you don't know.
Question: {query}

Context:
{ctx}

Return the answer and cite passages like [1], [2].
"""
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": "You answer strictly from the provided context."},
                {"role": "user", "content": prompt},
            ],
        )
        st.markdown("### Final Answer")
        st.write(resp.choices[0].message.content)
