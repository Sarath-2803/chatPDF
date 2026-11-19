import os
import tempfile
import traceback

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("GEMINI API key not found. Add GEMINI_API_KEY to your .env file.")
    st.stop()

st.set_page_config(
    page_title="Conversational PDF RAG",
    page_icon="📚",
    layout="wide"
)

st.title("Conversational PDF RAG")
st.caption("Upload PDFs and ask questions about their content.")


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.2
    )


embeddings = load_embeddings()
llm = load_llm()


if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_names" not in st.session_state:
    st.session_state.pdf_names = []


def get_response_text(response):

    content = response.content

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):

        text_parts = []

        for block in content:

            if isinstance(block, str):
                text_parts.append(block)

            elif isinstance(block, dict):

                text = block.get("text")

                if isinstance(text, str):
                    text_parts.append(text)

        result = "\n".join(text_parts)

        return result.strip() if isinstance(result, str) else ""

    return str(content).strip()


def process_pdfs(uploaded_files):

    all_documents = []

    for uploaded_file in uploaded_files:

        temp_path = None

        try:
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temp_file:

                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name

            loader = PyPDFLoader(temp_path)
            documents = loader.load()

            for document in documents:
                document.metadata["source"] = uploaded_file.name

            all_documents.extend(documents)

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    if not all_documents:
        raise ValueError("No readable text was found in the uploaded PDFs.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(all_documents)

    chunks = [
        chunk for chunk in chunks
        if isinstance(chunk.page_content, str)
        and chunk.page_content.strip()
    ]

    if not chunks:
        raise ValueError("No valid text chunks were generated from the PDFs.")

    # Build the new vectorstore before replacing the old one.
    new_vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="academic_documents"
    )

    return new_vectorstore


def get_chat_history():

    # Exclude the latest user question because it is
    # already stored in session state before generation.

    previous_messages = st.session_state.messages[:-1]

    history = []

    for message in previous_messages[-10:]:

        role = message["role"].upper()
        content = message["content"]

        if isinstance(content, str):
            history.append(f"{role}: {content}")

    return "\n".join(history)


def rewrite_question(question, chat_history):

    if not chat_history:
        return question

    prompt = f"""
        Given the conversation history and latest question,
        rewrite the latest question as a standalone question.

        Do not answer the question.
        Keep the original meaning.

        Conversation history:
        {chat_history}

        Latest question:
        {question}

        Standalone question:
    """

    response = llm.invoke(prompt)

    rewritten_question = get_response_text(response)

    if rewritten_question:
        return rewritten_question

    return question


def retrieve_documents(question):

    if st.session_state.vectorstore is None:
        return []

    documents = st.session_state.vectorstore.similarity_search(
        question,
        k=5
    )

    return documents


def generate_answer(question, documents, chat_history):

    if not documents:
        return "I couldn't find relevant information in the uploaded PDFs."

    context_parts = []

    for index, document in enumerate(documents, start=1):

        source = document.metadata.get("source", "Unknown PDF")
        page = document.metadata.get("page", "Unknown")

        context_parts.append(
            f"""
                [SOURCE {index}]
                    PDF: {source}
                    Page: {page}

                    Content:
                    {document.page_content}
            """
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
        You are a helpful PDF question-answering assistant.

        Answer the user's question using the provided PDF context.

        Rules:
        1. Use the PDF context as your primary source.
        2. Do not invent facts that are not in the context.
        3. If the answer is unavailable in the context, clearly say so.
        4. Keep the answer clear and relevant.
        5. Mention source numbers such as [SOURCE 1] when appropriate.

        Previous conversation:
        {chat_history if chat_history else "No previous conversation."}

        PDF context:
        {context}

        User question:
        {question}

        Answer:
    """

    response = llm.invoke(prompt)

    answer = get_response_text(response)

    if not answer:
        return "I couldn't generate a response. Please try again."

    return answer


st.sidebar.header("📄 Upload PDFs")

uploaded_files = st.sidebar.file_uploader(
    "Choose PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if st.sidebar.button("Process PDFs", type="primary"):

    if not uploaded_files:
        st.sidebar.warning("Please upload at least one PDF.")

    else:

        with st.spinner("Processing PDFs and creating embeddings..."):

            try:

                new_vectorstore = process_pdfs(uploaded_files)

                # Replace the old vectorstore only after
                # the new one has been created successfully.
                st.session_state.vectorstore = new_vectorstore

                st.session_state.pdf_names = [
                    file.name for file in uploaded_files
                ]

                # Reset conversation for the new documents.
                st.session_state.messages = []

                st.sidebar.success("PDFs processed successfully!")

            except Exception as error:

                st.sidebar.error(f"Processing failed: {error}")

                with st.expander("View error details"):
                    st.code(traceback.format_exc())


if st.session_state.pdf_names:

    st.sidebar.subheader("Indexed PDFs")

    for pdf_name in st.session_state.pdf_names:
        st.sidebar.write(f"📄 {pdf_name}")

    st.sidebar.caption(
        "Uploading and processing new PDFs replaces the previous index."
    )


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])



question = st.chat_input("Ask something about your PDFs...")

if question:

    if st.session_state.vectorstore is None:

        st.warning("Please upload and process PDFs first.")

    else:

        # Display and store the user's question.
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                try:

                    chat_history = get_chat_history()

                    standalone_question = rewrite_question(
                        question,
                        chat_history
                    )

                    documents = retrieve_documents(
                        standalone_question
                    )

                    answer = generate_answer(
                        question,
                        documents,
                        chat_history
                    )

                    st.markdown(answer)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as error:

                    st.error(f"Something went wrong: {error}")

                    with st.expander("View error details"):
                        st.code(traceback.format_exc())
