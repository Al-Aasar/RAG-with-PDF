# 📄 RAG with PDF

AI-powered chatbot that answers questions about your PDF documents.

🚀 **[Try it Live](https://rag-with-pdf-document.streamlit.app/)**

---

## What does it do?

Upload PDF files and ask any question about their content. The AI will read the documents and give you accurate answers.

---

## Features

- Upload multiple PDF files
- Ask questions in natural language
- Get AI-powered answers from your documents
- Free to use

---

## Technologies Used

- **Streamlit** - Web interface
- **Google Gemini** - AI for answering questions
- **HuggingFace** - Text processing
- **LangChain** - RAG framework

---

## How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/Al-Aasar/RAG-with-PDF.git
   cd rag-with-pdf
   ```

2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. Add your Google API key in `.streamlit/secrets.toml`:
   ```toml
   GOOGLE_API_KEY = "your-api-key-here"
   ```

4. Run the app:
   ```bash
   streamlit run app.py
   ```

---

## How to Use

1. Upload one or more PDF files
2. Click "Process PDFs"
3. Ask your question
4. Get your answer!

---

## Project Structure

```
├── app.py              # Main app
├── requirements.txt    # Dependencies
└── README.md          # This file
```

---

## 👨‍💻 Author

**Muhammad Al-Aasar**  
🎓 B.Sc. in Computer Science, Tanta University  
🔗 [LinkedIn Profile](https://www.linkedin.com/in/muhammad-al-aasar-455b78329)  
📞 +20 1015088811

---

⭐ Give this project a star if you find it useful!
