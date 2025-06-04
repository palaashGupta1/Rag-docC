import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

# Load API Key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="RAG Chatbot", layout="wide")

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

uploaded_file = st.file_uploader("Upload your document (PDF only)", type="pdf")

if uploaded_file:
    with open(f"docs/{uploaded_file.name}", "wb") as f:
        f.write(uploaded_file.getbuffer())

    loader = PyPDFLoader(f"docs/{uploaded_file.name}")
    documents = loader.load()

    # Split & embed
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)

    retriever = vectorstore.as_retriever()

    def custom_filtering_qa(question):
        result = retriever.get_relevant_documents(question)
        if not result:
            return "🤖 Sorry, I can't answer that. Please ask a question related to the document."
        qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(temperature=0, openai_api_key=openai_api_key),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False
        )
        return qa.run(question)

    st.markdown("### Ask a question based on the document")

    query = st.text_input("Your question:")

    if query:
        response = custom_filtering_qa(query)
        st.markdown(f"**Response:** {response}")
