# ResearchMate AI – Research Paper Question Answering System

ResearchMate AI is an intelligent research paper analysis assistant that allows users to upload research papers in PDF format and ask questions about their contents. It uses **Retrieval-Augmented Generation (RAG)** to generate context-based answers about methodology, datasets, results, limitations, and other research-related topics.

The system retrieves relevant sections from the uploaded paper before generating an answer, helping keep responses grounded in the source document. Answers include source page references for verification.

## Key Features

- Upload research papers in PDF format.
- Ask custom questions about the uploaded paper.
- Preset question buttons for Methodology, Dataset, Results, and Limitations.
- Semantic search using FAISS vector database.
- Context-aware answer generation using Groq LLM.
- Source page references for answer verification.
- Responses restricted to retrieved paper content.
- Handles questions whose answers are not available in the document.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| User Interface | Streamlit |
| RAG Framework | LangChain |
| PDF Processing | PyPDFLoader |
| Text Splitting | Recursive Character Text Splitter |
| Embedding Model | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Database | FAISS |
| Language Model | Groq – Llama 3.3 70B Versatile |

## RAG Architecture and Workflow

1. **PDF Upload:** The user uploads a research paper through the Streamlit interface.
2. **Document Extraction:** PyPDFLoader extracts text and page metadata from the PDF.
3. **Text Chunking:** The extracted text is divided into smaller overlapping chunks.
4. **Embedding Generation:** HuggingFace generates vector embeddings for each text chunk.
5. **Vector Storage:** FAISS stores the embeddings for efficient similarity-based retrieval.
6. **Question Retrieval:** The user's question is converted into an embedding, and the most relevant chunks are retrieved.
7. **Answer Generation:** The retrieved context is passed to the Groq LLM along with a grounding prompt.
8. **Source References:** The application displays the answer with relevant source page numbers.

## Project Structure

```text
ResearchMate-AI/
│
├── app.py              # Streamlit user interface
├── rag_pipeline.py     # RAG pipeline implementation
├── requirements.txt    # Project dependencies
├── .env.example        # API key template
├── .gitignore          # Git ignored files
└── README.md           # Project documentation
```

## Installation and Setup (Windows)

### 1. Prerequisites

- Python 3.10, 3.11, or 3.12 (64-bit recommended)
- Visual Studio Code
- Internet connection
- Groq API key

### 2. Clone the Repository

```powershell
git clone https://github.com/noamandesai/ResearchMate-AI.git
cd ResearchMate-AI
```

### 3. Create and Activate Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

### 5. Configure Groq API Key

Obtain an API key from [Groq Console](https://console.groq.com/keys).

Create a `.env` file in the project root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Replace the placeholder with your actual API key. Never upload the `.env` file to GitHub.

### 6. Run the Application

```powershell
streamlit run app.py
```

Open the following address in your browser:

http://localhost:8501

## How to Use

1. Launch ResearchMate AI.
2. Upload a research paper in PDF format.
3. Wait until document processing and indexing are completed.
4. Select one of the preset questions:
   - Methodology
   - Dataset
   - Results
   - Limitations
5. Alternatively, enter a custom question in the chat interface.
6. Review the generated answer and its source page references.

## Limitations

- Scanned or image-only PDFs are not supported because text extraction requires selectable text.
- Answer quality depends on the quality and content of the uploaded research paper.
- Internet access is required for Groq API requests and the initial embedding model download.
- The system may not provide an answer when relevant information is absent from the retrieved document context.

## Troubleshooting

| Issue | Solution |
|---|---|
| Missing GROQ_API_KEY | Verify that `.env` exists and contains a valid API key. |
| Import errors | Activate the virtual environment and reinstall dependencies. |
| FAISS installation errors | Use a compatible 64-bit Python version (3.10–3.12). |
| Model/API errors | Check the API key and verify that the configured Groq model is supported. |
| Application not opening | Confirm that Streamlit is installed and the application is running on port 8501. |

## Security

- API keys are stored locally in `.env`.
- `.env` is excluded from Git tracking using `.gitignore`.
- Never publish API keys or other sensitive credentials.

## Author

**Noaman M. Desai**

ResearchMate AI – GenAI Individual Assignment

GitHub: https://github.com/noamandesai/ResearchMate-AI