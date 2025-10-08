import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# --- App config ---
st.set_page_config(page_title="PDF RAG Chat", page_icon="💬", layout="wide")

# --- Secrets / API key ---
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# --- Session state init ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Upload PDFs from the sidebar, click Process, then ask me anything about them."}
    ]

if "vector_store_ready" not in st.session_state:
    st.session_state.vector_store_ready = False

if "vs" not in st.session_state:
    st.session_state.vs = None

# --- Core helpers ---
def extract_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
    return text

def chunk_text(text, chunk_size=1000, overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)

def build_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.from_texts(chunks, embedding=embeddings)
    return vs

def qa_chain():
    prompt_template = """
    You are a helpful assistant answering strictly from the provided context.
    If the answer is not in the context, say: "The answer is not available in the context."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain

def answer_question(question, k=4):
    # Retrieve relevant docs
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    docs = st.session_state.vs.similarity_search(question, k=k)
    # Run QA chain
    chain = qa_chain()
    out = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    return out["output_text"]

# --- Sidebar: documents + controls ---
with st.sidebar:
    st.title("📁 Documents")
    pdfs = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    if st.button("Process PDFs", use_container_width=True):
        if not pdfs:
            st.warning("Please upload at least one PDF.")
        else:
            with st.spinner("Reading and indexing your PDFs..."):
                raw = extract_pdf_text(pdfs)
                chunks = chunk_text(raw, 1000, 200)
                st.session_state.vs = build_vector_store(chunks)
                st.session_state.vector_store_ready = True
            st.success("✅ Ready! Start chatting from the box below.")
    st.divider()
    if st.button("Clear chat", type="secondary", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Chat cleared. Ask anything about your processed PDFs."}
        ]
    if st.button("Reset index", type="secondary", use_container_width=True):
        st.session_state.vs = None
        st.session_state.vector_store_ready = False
        st.success("Index cleared. Re-upload and process PDFs to chat again.")

# --- Chat history render ---
st.header("💬 Chat with your PDFs")
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# --- Chat input ---
prompt = st.chat_input("Ask about your PDFs...")
if prompt:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Guard: index ready?
    if not st.session_state.vector_store_ready or st.session_state.vs is None:
        msg = "Please upload PDFs and click 'Process PDFs' first."
        with st.chat_message("assistant"):
            st.markdown(msg)
        st.session_state.messages.append({"role": "assistant", "content": msg})
    else:
        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = answer_question(prompt, k=4)
                except Exception as e:
                    response = f"Sorry, something went wrong: {str(e)}"
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
