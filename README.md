# DocuChat-RAG (Simple PDF Q&A)

Upload PDFs → builds a local FAISS index (MiniLM embeddings) → ask questions.
- Works **without** OpenAI key (shows top passages).
- With `OPENAI_API_KEY` → generates a final answer using the retrieved context.

## Run
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1       # Windows PowerShell
pip install -r requirements.txt
setx OPENAI_API_KEY "sk-..."       # or set env in PyCharm Run Config
streamlit run app.py
