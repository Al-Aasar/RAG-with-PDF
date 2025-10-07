import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate


os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, 
        chunk_overlap=1000
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    Make sure to provide all the details. If the answer is not in the provided context, 
    just say "The answer is not available in the context", don't provide wrong answers.
    
    Context:\n{context}\n
    Question:\n{question}\n
    
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.3
    )
    
    prompt = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )
    
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    new_db = FAISS.load_local(
        "faiss_index", 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    docs = new_db.similarity_search(user_question)
    
    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    
    return response["output_text"]


def main():
    st.set_page_config(
        page_title="PDF Chat with Gemini",
        page_icon="📄",
        layout="wide"
    )
    
    st.header("📄 Chat with PDF using Google Gemini")
    st.write("Upload your PDF files and ask questions about their content!")
    

    with st.sidebar:
        st.title("📁 Upload Documents")
        pdf_docs = st.file_uploader(
            "Upload PDF Files", 
            accept_multiple_files=True,
            type=['pdf']
        )
        
        if st.button("Process PDFs"):
            if pdf_docs:
                with st.spinner("Processing your PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("✅ PDFs processed successfully! You can now ask questions.")
            else:
                st.warning("Please upload at least one PDF file.")
    

    user_question = st.text_input("Ask a question about your PDFs:")
    
    if user_question:
        if os.path.exists("faiss_index"):
            with st.spinner("Generating answer..."):
                response = user_input(user_question)
                st.write("### Answer:")
                st.write(response)
        else:
            st.warning("⚠️ Please upload and process PDF files first!")
    

    with st.expander("ℹ️ How to use"):
        st.write("""
        1. Upload one or more PDF files using the sidebar
        2. Click 'Process PDFs' button to analyze the documents
        3. Ask any question about the content of your PDFs
        4. Get AI-powered answers based on the document context
        """)

if __name__ == "__main__":
    main()
