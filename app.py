import streamlit as st
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ============================
# PAGE CONFIG
# ============================
st.set_page_config(page_title="RAG PDF Chatbot", page_icon="🤖", layout="wide")

st.markdown("""
    <style>
        .block-container { padding-top: 20px; }
        textarea, input {
            background-color: #222 !important;
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 RAG Chatbot PDF (Streamlit Cloud + Groq API – FREE)")

# ============================
# SIDEBAR UPLOAD
# ============================
st.sidebar.header("📄 Upload PDF")
uploaded_files = st.sidebar.file_uploader(
    "Upload một hoặc nhiều file PDF",
    type=["pdf"],
    accept_multiple_files=True
)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Lưu file PDF vào thư mục máy chủ
if uploaded_files:
    for file in uploaded_files:
        file_path = os.path.join(DATA_DIR, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
    st.sidebar.success("✔ File đã được lưu vào thư mục data/")

# ============================
# BUILD INDEX (CACHE)
# ============================

@st.cache_resource
def build_index():
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]

    if not pdf_files:
        return None

    st.sidebar.info("🔄 Đang load tài liệu...")

    # đọc tài liệu
    docs = SimpleDirectoryReader(DATA_DIR).load_data()

    # LLM Groq miễn phí
    llm = Groq(
        model="llama3-8b-8192",
        api_key=st.secrets["GROQ_API_KEY"]
    )

    # Embedding miễn phí HuggingFace
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # tạo index
    index = VectorStoreIndex.from_documents(
        docs,
        llm=llm,
        embed_model=embed_model
    )
    return index

index = build_index()

# ============================
# CHAT ENGINE
# ============================
if index:
    chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=False)
else:
    chat_engine = None

# ============================
# CHAT UI
# ============================
st.subheader("💬 Chat với PDF của bạn")

if "messages" not in st.session_state:
    st.session_state.messages = []

# hiển thị tin nhắn cũ
for role, msg in st.session_state.messages:
    st.chat_message(role).markdown(msg)

# input chat
user_input = st.chat_input("Nhập câu hỏi...")

if user_input:
    st.session_state.messages.append(("user", user_input))

    if chat_engine:
        with st.spinner("🤖 Đang suy nghĩ..."):
            response = chat_engine.chat(user_input)
            bot_reply = response.response
            st.session_state.messages.append(("assistant", bot_reply))
    else:
        st.session_state.messages.append(("assistant", "⚠ Hãy upload ít nhất 1 PDF."))

    st.rerun()
