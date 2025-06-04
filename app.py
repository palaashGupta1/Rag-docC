import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# Load environment
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Streamlit UI config
st.set_page_config(page_title="Pastel RAG Chatbot", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #fef6e4; }
    .stTextInput, .stTextArea, .stButton, .stFileUploader { background-color: #f3d2c1; }
    .css-2trqyj { background-color: #f9dcc4; }
    </style>
""", unsafe_allow_html=True)

st.title("📄 Pastel RAG Chatbot")
st.markdown("Upload a **PDF**, and ask questions based **only** on its content.")

# File upload
uploaded_file = st.file_uploader("Upload your document (PDF only)", type="pdf")

if uploaded_file:
    # Save file
    if not os.path.exists("docs"):
        os.makedirs("docs")
    file_path = os.path.join("docs", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load + split document
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)
    docs = docs[:100]  # Optional: prevent overload

    # Embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    try:
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    except Exception as e:
        st.error("❌ Failed to process document. Try a smaller or simpler PDF.")
        st.stop()

    # Answer function
    def custom_rag_answer(query):
        matched_docs = retriever.get_relevant_documents(query)

        # If top result is weak, reject (rough heuristic)
        if not matched_docs or len(matched_docs[0].page_content.strip()) < 30:
            return "🤖 I can’t answer that. Please ask a question related to the uploaded document."

        # Proceed with QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(temperature=0, openai_api_key=openai_api_key),
            chain_type="stuff",
            retriever=retriever
        )
        return qa_chain.run(query)

    # User question UI
    st.markdown("### Ask a question about your PDF:")
    user_question = st.text_input("Your question:")

    if user_question:
        with st.spinner("💡 Thinking..."):
            response = custom_rag_answer(user_question)
        st.success("✅ Answer:")
        st.markdown(response)
