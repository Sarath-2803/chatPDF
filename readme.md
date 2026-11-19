# chatPDF

A Streamlit application for chatting with uploaded PDF documents. It extracts PDF text, creates searchable embeddings, and uses Google Gemini to answer questions with conversational context.

## Features

- Upload and process one or more PDF files
- Semantic search using Hugging Face `all-MiniLM-L6-v2` embeddings
- Local Chroma vector store for indexed document chunks
- Conversational follow-up questions with question rewriting
- Source and page-aware answers based on retrieved PDF content

## Requirements

- Python 3.10 or newer
- A Google Gemini API key

## Installation

```bash
git clone https://github.com/Sarath-2803/chatPDF.git
cd chatPDF
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
```

## Run the application

```bash
streamlit run app.py
```

Open the local URL shown by Streamlit, upload your PDFs from the sidebar, and select **Process PDFs** before asking questions.

## How it works

1. Uploaded PDFs are read and split into overlapping text chunks.
2. The chunks are embedded with `sentence-transformers/all-MiniLM-L6-v2`.
3. Chroma retrieves the most relevant chunks for each question.
4. Gemini generates an answer using the retrieved PDF context and recent conversation history.

Processing a new set of PDFs replaces the previous document index and resets the conversation.

## Project structure

```text
chatPDF/
├── app.py
├── requirements.txt
├── .env                 # Local only; do not commit
└── readme.md
```

## Notes

- Never commit your `.env` file or API keys.
- The first run may download the embedding model.
- Scanned PDFs without selectable text may not produce usable answers unless OCR text is available.
