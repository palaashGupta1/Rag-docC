import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# Load environment variable
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set Streamlit layout
st.set_page_config(page_title="Pastel RAG Chatbot", layout="wide")

st.markdown(
    """
    <style>
    .main { background-color: #fef6e4; }
    .stTextInput, .stTextArea, .stButton, .stFileUploader { background-color: #f3d2c1; }
    .css-2trqyj { background-color: #f9dcc4; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📄 Pastel RAG Chatbot")
st.markdown("Upload a **PDF**, and ask questions based **only** on its content.")

# Upload block
uploaded_file = st.file_uploader("Upload your document (PDF only)", type="pdf")

if uploaded_file:
    # Ensure docs folder exists
    if not os.path.exists("docs"):
        os.makedirs("docs")

    # Save file
    file_path = os.path.join("docs", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load and chunk the PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)

    # Embedding and retrieval setup
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()

    def custom_rag_answer(query):
        relevant_docs = retriever.get_relevant_documents(query)

        # Very basic check: require at least 2 chunks returned
        if not relevant_docs or len(relevant_docs) < 2:
            return "🤖 Sorry, I couldn't find relevant information in your uploaded document."

        # Answer using LLM constrained by retriever
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(temperature=0, openai_api_key=openai_api_key),
            chain_type="stuff",
            retriever=retriever
        )
        return qa_chain.run(query)

    # UI: user input and answer
    st.markdown("### Ask a question about your PDF:")
    user_question = st.text_input("Your question:")

    if user_question:
        with st.spinner("🧠 Thinking..."):
            response = custom_rag_answer(user_question)
        st.success("✅ Answer:")
        st.markdown(response)
