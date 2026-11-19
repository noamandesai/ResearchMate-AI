"""ResearchMate AI - clean, minimal Streamlit interface."""

import html

import streamlit as st

from rag_pipeline import (
    NOT_FOUND_MESSAGE,
    PRESET_QUESTIONS,
    RAGError,
    answer_question,
    build_vectorstore,
)

st.set_page_config(
    page_title="ResearchMate AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- Visual design ----------------
st.markdown(
    """
    <style>
    :root {
        --page: #f5f7f6;
        --surface: #ffffff;
        --surface-soft: #eef3f0;
        --ink: #1d2924;
        --muted: #697770;
        --line: #dce5df;
        --accent: #28634f;
        --accent-hover: #1f503f;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: var(--page);
        color: var(--ink);
    }

    [data-testid="stHeader"] {
        background: rgba(245, 247, 246, 0.92);
    }

    .block-container {
        max-width: 1080px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background: #edf2ef;
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }

    h1, h2, h3, p, label, .stMarkdown {
        color: var(--ink);
    }

    .brand-kicker {
        color: var(--accent);
        font-size: 0.76rem;
        font-weight: 750;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 0.55rem;
    }

    .hero-title {
        color: var(--ink);
        font-size: clamp(2rem, 4vw, 3rem);
        font-weight: 780;
        letter-spacing: -0.055em;
        line-height: 1.08;
        margin: 0;
    }

    .hero-subtitle {
        color: var(--muted);
        font-size: 1.02rem;
        margin-top: 0.7rem;
        margin-bottom: 1.7rem;
    }

    .section-label {
        color: #52645a;
        font-size: 0.78rem;
        font-weight: 750;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin: 0.2rem 0 0.8rem 0;
    }

    .side-heading {
        color: var(--ink);
        font-size: 1.15rem;
        font-weight: 750;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }

    .side-description {
        color: var(--muted);
        font-size: 0.88rem;
        margin-bottom: 1.2rem;
    }

    .status-card {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 1rem;
        margin-top: 0.7rem;
        color: var(--ink);
        overflow-wrap: anywhere;
    }

    .status-name {
        color: var(--ink);
        font-size: 0.92rem;
        font-weight: 700;
        margin-bottom: 0.65rem;
    }

    .status-meta {
        color: var(--muted);
        font-size: 0.82rem;
        line-height: 1.8;
    }

    .ready-pill {
        display: inline-block;
        color: #245c47;
        background: #e5f2eb;
        border: 1px solid #c8e2d2;
        border-radius: 999px;
        padding: 0.22rem 0.6rem;
        font-size: 0.72rem;
        font-weight: 750;
        margin-bottom: 0.75rem;
    }

    /* Keep every button readable, including Streamlit's secondary buttons. */
    div.stButton > button,
    div.stDownloadButton > button {
        width: 100%;
        min-height: 2.7rem;
        color: #244d3e !important;
        background: #ffffff !important;
        border: 1px solid #cbd9d0 !important;
        border-radius: 11px !important;
        font-weight: 650 !important;
        box-shadow: none !important;
        transition: all 150ms ease;
    }

    div.stButton > button:hover,
    div.stDownloadButton > button:hover {
        color: #ffffff !important;
        background: var(--accent) !important;
        border-color: var(--accent) !important;
    }

    div.stButton > button:focus,
    div.stDownloadButton > button:focus {
        color: #ffffff !important;
        background: var(--accent) !important;
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px rgba(40, 99, 79, 0.16) !important;
    }

    div.stButton > button:active {
        background: var(--accent-hover) !important;
    }

    [data-testid="stFileUploader"] {
        background: #ffffff;
        border: 1px dashed #b9cabe;
        border-radius: 13px;
        padding: 0.45rem;
    }

    [data-testid="stFileUploader"] section {
        background: transparent;
    }

    [data-testid="stFileUploader"] button {
        color: #244d3e !important;
        background: #edf4ef !important;
        border: 1px solid #cbd9d0 !important;
        border-radius: 8px !important;
    }

    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] span {
        color: #52645a !important;
    }

    [data-testid="stChatInput"] {
        border: 1px solid #cbd9d0 !important;
        border-radius: 15px !important;
        background: #ffffff !important;
    }

    [data-testid="stChatInput"] textarea {
        color: var(--ink) !important;
    }

    [data-testid="stChatMessage"] {
        background: #ffffff;
        border: 1px solid #e1e8e3;
        border-radius: 15px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.75rem;
    }

    [data-testid="stChatMessage"] p {
        color: var(--ink);
    }

    [data-testid="stCaptionContainer"] p {
        color: var(--muted) !important;
    }

    hr {
        border-color: var(--line);
        margin: 1.35rem 0;
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    @media (max-width: 700px) {
        .block-container {
            padding: 1.3rem 1rem 2rem 1rem;
        }
        .hero-title {
            font-size: 2.1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------- Session state ----------------
def init_state():
    defaults = {
        "messages": [],
        "vectorstore": None,
        "doc_name": None,
        "doc_stats": None,
        "doc_key": None,
        "uploader_key": 0,
        "pending_question": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_all():
    st.session_state.messages = []
    st.session_state.vectorstore = None
    st.session_state.doc_name = None
    st.session_state.doc_stats = None
    st.session_state.doc_key = None
    st.session_state.pending_question = None
    st.session_state.uploader_key += 1


def render_message(msg):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("pages"):
            page_list = ", ".join(str(page) for page in msg["pages"])
            st.caption(f"Source pages: {page_list}")


init_state()


# ---------------- Sidebar: document workspace ----------------
with st.sidebar:
    st.markdown('<div class="side-heading">📚 ResearchMate AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="side-description">Your personal research paper assistant</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-label">Your document</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload a research paper",
        type=["pdf"],
        key=f"uploader_{st.session_state.uploader_key}",
        help="Choose a PDF file to start asking questions.",
    )

    if uploaded is not None:
        doc_key = f"{uploaded.name}-{uploaded.size}"
        if st.session_state.doc_key != doc_key:
            st.session_state.messages = []
            st.session_state.vectorstore = None
            st.session_state.doc_stats = None
            try:
                with st.spinner("Reading and indexing your paper..."):
                    vectorstore, stats = build_vectorstore(uploaded.getvalue())

                st.session_state.vectorstore = vectorstore
                st.session_state.doc_stats = stats
                st.session_state.doc_name = uploaded.name
                st.session_state.doc_key = doc_key
            except RAGError as exc:
                st.session_state.doc_key = None
                st.session_state.doc_name = None
                st.error(str(exc))
            except Exception as exc:
                st.session_state.doc_key = None
                st.session_state.doc_name = None
                st.error(f"Unable to process this PDF: {exc}")

    if st.session_state.vectorstore is not None:
        stats = st.session_state.doc_stats or {}
        safe_name = html.escape(st.session_state.doc_name or "Research paper")
        st.markdown(
            f"""
            <div class="status-card">
                <div class="ready-pill">● DOCUMENT READY</div>
                <div class="status-name">{safe_name}</div>
                <div class="status-meta">
                    {stats.get("pages", "—")} pages with text<br>
                    {stats.get("chunks", "—")} searchable sections
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.caption("Your uploaded PDF and its processing status will appear here.")

    st.divider()
    if st.button("＋ Start with a new paper", use_container_width=True):
        clear_all()
        st.rerun()

    st.caption("Your questions are answered using the uploaded document.")


# ---------------- Main workspace ----------------
st.markdown('<div class="brand-kicker">AI-powered paper companion</div>', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">Understand the paper.<br>Skip the noise.</h1>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Ask focused questions and explore the key ideas in your research paper.</div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="section-label">Explore your paper</div>', unsafe_allow_html=True)

quick_questions = list(PRESET_QUESTIONS.items())
icons = {"Method": "🧪", "Dataset": "🗂️", "Results": "📊", "Limitations": "⚠️"}
cols = st.columns(len(quick_questions)) if quick_questions else []

for col, (label, question) in zip(cols, quick_questions):
    with col:
        if st.button(
            f"{icons.get(label, '✦')}  {label}",
            key=f"preset_{label}",
            use_container_width=True,
        ):
            st.session_state.pending_question = question

st.divider()

if not st.session_state.messages:
    if st.session_state.vectorstore is None:
        st.markdown("### Start with a research paper")
        st.caption(
            "Upload a PDF from the left panel. Once it is ready, choose one of the prompts above "
            "or ask your own question below."
        )
    else:
        st.markdown("### Your paper is ready")
        st.caption(
            "Try asking about the research method, dataset, findings, limitations, "
            "or any detail you want to understand."
        )

# Existing conversation
for message in st.session_state.messages:
    render_message(message)

# A preset prompt is handled on the next rerun; otherwise accept typed chat input.
typed_question = st.chat_input("Ask anything about your research paper...")
question = st.session_state.pending_question or typed_question
st.session_state.pending_question = None

if question:
    if st.session_state.vectorstore is None:
        st.warning("Upload a research paper PDF from the left panel before asking a question.")
    else:
        user_message = {"role": "user", "content": question}
        st.session_state.messages.append(user_message)
        render_message(user_message)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Finding relevant sections in your paper..."):
                    answer, pages = answer_question(st.session_state.vectorstore, question)

                if NOT_FOUND_MESSAGE in answer:
                    pages = []

                st.markdown(answer)
                if pages:
                    st.caption("Source pages: " + ", ".join(str(page) for page in pages))

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "pages": pages}
                )
            except RAGError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Something went wrong: {exc}")
