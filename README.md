# ResearchMate AI

Intelligent Research Paper Analysis Assistant. Upload a research paper (PDF) and ask about its
methodology, dataset, results and limitations. Answers come only from retrieved parts of the paper
(RAG with FAISS + Groq), with source page numbers.

## Project structure
```
app.py            Streamlit interface
rag_pipeline.py   PDF loading, chunking, embeddings, FAISS retrieval, Groq LLM
requirements.txt  Dependencies
.env.example      API key placeholder
.gitignore
README.md
```

## Setup on Windows (VS Code)

1. **Install Python 3.10, 3.11 or 3.12** from python.org (tick "Add Python to PATH"). Check: `python --version`
2. **Get a free Groq API key** at https://console.groq.com/keys
3. Open the `researchmate-ai` folder in VS Code (File > Open Folder), then open a terminal (Ctrl + `).
4. **Create and activate a virtual environment** (PowerShell):
   ```powershell
   python -m venv venv
   venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks the script, run once:
   `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or use Command Prompt: `venv\Scripts\activate.bat`
5. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```
6. **Add your API key**
   ```powershell
   copy .env.example .env
   ```
   Open `.env` and replace `your_groq_api_key_here` with your real key.
7. **Run**
   ```powershell
   streamlit run app.py
   ```
   The app opens at http://localhost:8501.

## How to use
1. Upload a PDF in the sidebar and wait for "Ready for questions".
2. Click **Method**, **Dataset**, **Results** or **Limitations**, or type your own question.
3. Use **Clear chat & upload new paper** to start over.

## Notes
- The first run downloads the embedding model (`all-MiniLM-L6-v2`, ~90 MB). Internet is required once.
- Scanned (image-only) PDFs are not supported because they contain no extractable text.
- If Groq retires the default model, set `GROQ_MODEL` in `.env` to a currently supported one.
- Never commit `.env`; it is already listed in `.gitignore`.

## Troubleshooting
- **GROQ_API_KEY is missing**: make sure `.env` exists (not `.env.example`) and restart Streamlit.
- **Import errors**: confirm the virtual environment is active, then re-run `pip install -r requirements.txt`.
- **faiss install fails**: use Python 3.10-3.12 (64-bit).
