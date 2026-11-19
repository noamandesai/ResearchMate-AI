"""RAG pipeline for ResearchMate AI.

Handles PDF loading, chunking, embeddings, FAISS retrieval and Groq LLM calls.
"""

import os
import tempfile
from functools import lru_cache

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 5

NOT_FOUND_MESSAGE = "This information is not mentioned in the uploaded research paper."

SYSTEM_PROMPT = (
    "You are a research paper analysis assistant. Answer questions strictly using the "
    "retrieved context from the uploaded research paper. Do not invent facts, datasets, "
    "results, metrics or methodologies. If the requested information is unavailable in the "
    "retrieved context, clearly state that it is not mentioned in the uploaded paper. "
    "Provide concise, accurate and well-structured answers. Include source page references "
    "whenever available."
)

PRESET_QUESTIONS = {
    "Method": "What methodology, algorithms and techniques are used in this research paper?",
    "Dataset": "Which dataset is used? Explain its size, features and source if mentioned.",
    "Results": "What are the key findings, performance metrics and experimental results?",
    "Limitations": "What limitations, challenges and future work are discussed?",
}


class RAGError(Exception):
    """User-friendly error raised by the pipeline."""


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the embedding model once (downloaded on first run)."""
    try:
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as exc:
        raise RAGError(
            f"Could not load the embedding model. Check your internet connection "
            f"for the first run. Details: {exc}"
        ) from exc


def build_vectorstore(pdf_bytes: bytes):
    """Extract text from a PDF, chunk it, embed it and index it in FAISS.

    Returns (vectorstore, stats_dict).
    """
    if not pdf_bytes:
        raise RAGError("The uploaded file is empty.")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        try:
            pages = PyPDFLoader(tmp_path).load()
        except Exception as exc:
            raise RAGError(
                "Could not read this PDF. It may be corrupted or password-protected."
            ) from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    pages = [p for p in pages if p.page_content and p.page_content.strip()]
    if not pages:
        raise RAGError(
            "No readable text was found in this PDF. It may be a scanned document "
            "(image only), which is not supported."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(pages)
    if not chunks:
        raise RAGError("The document could not be split into text chunks.")

    try:
        vectorstore = FAISS.from_documents(chunks, get_embeddings())
    except RAGError:
        raise
    except Exception as exc:
        raise RAGError(f"Failed to build the vector index: {exc}") from exc

    return vectorstore, {"pages": len(pages), "chunks": len(chunks)}


def _get_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key or api_key == "your_groq_api_key_here":
        raise RAGError(
            "GROQ_API_KEY is missing. Copy `.env.example` to `.env`, add your key, "
            "and restart the app."
        )
    model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL).strip() or DEFAULT_GROQ_MODEL
    return ChatGroq(model=model, api_key=api_key, temperature=0)


def answer_question(vectorstore, question: str):
    """Retrieve relevant chunks and answer strictly from them.

    Returns (answer_text, sorted_list_of_page_numbers).
    """
    question = (question or "").strip()
    if not question:
        raise RAGError("Please enter a question.")
    if vectorstore is None:
        raise RAGError("Please upload a research paper first.")

    llm = _get_llm()

    try:
        docs = vectorstore.similarity_search(question, k=TOP_K)
    except Exception as exc:
        raise RAGError(f"Retrieval failed: {exc}") from exc

    if not docs:
        return NOT_FOUND_MESSAGE, []

    context_parts = []
    pages = set()
    for doc in docs:
        page = doc.metadata.get("page")
        label = f"Page {page + 1}" if isinstance(page, int) else "Page unknown"
        if isinstance(page, int):
            pages.add(page + 1)
        context_parts.append(f"[{label}]\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)

    user_prompt = (
        f"Retrieved context from the uploaded research paper:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer using only the context above. Use short headings or bullet points "
        f"where helpful and cite pages like (Page 4). If the context does not contain "
        f"the answer, reply exactly: \"{NOT_FOUND_MESSAGE}\""
    )

    try:
        response = llm.invoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_prompt)]
        )
    except Exception as exc:
        msg = str(exc)
        low = msg.lower()
        if "401" in msg or "invalid api key" in low or "authentication" in low:
            raise RAGError("The Groq API key was rejected. Please check your key.") from exc
        if "429" in msg or "rate limit" in low:
            raise RAGError("Groq rate limit reached. Wait a moment and try again.") from exc
        if "model" in low and ("decommissioned" in low or "not found" in low):
            raise RAGError(
                "The configured Groq model is unavailable. Set GROQ_MODEL in `.env` "
                "to a currently supported model."
            ) from exc
        raise RAGError(f"The Groq API request failed: {msg}") from exc

    answer = (response.content or "").strip() or NOT_FOUND_MESSAGE
    return answer, sorted(pages)
