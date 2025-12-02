import streamlit as st
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface.base import HuggingFaceEmbedding

# ============================
# PAGE CONFIG
# ============================
st.set_page_config(page_title="RAG Chatbot PDF", page_icon="🤖", layout="wide")
st.title("🤖 RAG Chatbot PDF – Streamlit Cloud + Groq API (FREE)")

# ============================
# UPLOAD PDF
# ============================
st.sidebar.header("📄 Upload PDF")
uploaded_files = st.sidebar.file_uploader(
    "Tải lên một hoặc nhiều file PDF",
    type=["pdf"],
    accept_multiple_files=True
)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Lưu file PDF vào thư mục data/
if uploaded_files:
    for file in uploaded_files:
        file_path = os.path.join(DATA_DIR, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
    st.sidebar.success("✔ PDF đã lưu vào thư mục data/")

# ============================
# BUILD INDEX (CACHE)
# ============================
@st.cache_resource
def load_index():
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        return None

    st.sidebar.info("🔄 Đang xử lý tài liệu...")

    # Load PDF content
    docs = SimpleDirectoryReader(DATA_DIR).load_data()

    # FREE Embedding
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # FREE Groq LLM
    llm = Groq(
        model="llama3-8b-8192",
        api_key=st.secrets["GROQ_API_KEY"]
    )

    index = VectorStoreIndex.from_documents(
        docs,
        llm=llm,
        embed_model=embed_model
    )

    return index


index = load_index()

# ============================
# CHAT ENGINE
# ============================
if index:
    # MUST PASS LLM HERE TO FIX "default llm" ERROR
    chat_engine = index.as_chat_engine(
        llm=Groq(
            model="llama3-8b-8192",
            api_key=st.secrets["GROQ_API_KEY"]
        ),
        chat_mode="condense_question",
        verbose=False
    )
else:
    chat_engine = None

# ============================
# CHAT UI
# ============================
st.subheader("💬 Chat với tài liệu PDF của bạn")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show old messages
for role, text in st.session_state.messages:
    st.chat_message(role).markdown(text)

# Chat input
user_message = st.chat_input("Nhập câu hỏi...")

if user_message:
    st.session_state.messages.append(("user", user_message))

    if chat_engine:
        with st.spinner("🤖 Đang suy nghĩ..."):
            response = chat_engine.chat(user_message)
            bot_reply = response.response
            st.session_state.messages.append(("assistant", bot_reply))
    else:
        st.session_state.messages.append(("assistant", "⚠ Vui lòng upload ít nhất 1 PDF."))

    st.rerun()
