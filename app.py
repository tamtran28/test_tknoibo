import streamlit as st
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface.base import HuggingFaceEmbedding

# ============================
# PAGE CONFIG
# ============================
st.set_page_config(page_title="RAG Chatbot PDF", page_icon="🤖", layout="wide")
st.title("🤖 RAG Chatbot PDF – Streamlit Cloud + Groq (FREE)")

# ============================
# UPLOAD PDF
# ============================
st.sidebar.header("📄 Upload PDF")
uploaded_files = st.sidebar.file_uploader(
    "Tải lên file PDF",
    type=["pdf"],
    accept_multiple_files=True
)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Lưu file vào thư mục data/
if uploaded_files:
    for file in uploaded_files:
        file_path = os.path.join(DATA_DIR, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
    st.sidebar.success("✔ PDF đã lưu vào data/")

# ============================
# BUILD INDEX (CACHE)
# ============================
@st.cache_resource
def build_index():
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        return None

    st.sidebar.info("🔄 Đang xử lý tài liệu...")

    # Load PDF thành văn bản
    documents = SimpleDirectoryReader(DATA_DIR).load_data()

    # Embedding miễn phí HuggingFace
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # LLM miễn phí của Groq — MODEL MỚI
    llm = Groq(
        model="llama3-8b",   # bạn có thể đổi thành "llama3-70b"
        api_key=st.secrets["GROQ_API_KEY"]
    )

    index = VectorStoreIndex.from_documents(
        documents,
        llm=llm,
        embed_model=embed_model
    )

    return index


index = build_index()

# ============================
# CHAT ENGINE
# ============================
if index:
    llm_chat = Groq(
        model="llama3-8b",  # Hoặc "llama3-70b" nếu muốn trả lời mạnh hơn
        api_key=st.secrets["GROQ_API_KEY"]
    )

    chat_engine = index.as_chat_engine(
        llm=llm_chat,
        chat_mode="condense_question",
        verbose=False
    )
else:
    chat_engine = None

# ============================
# CHAT UI
# ============================
st.subheader("💬 Chat với PDF")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử hội thoại
for role, text in st.session_state.messages:
    st.chat_message(role).markdown(text)

# Nhập câu hỏi
user_input = st.chat_input("Nhập câu hỏi...")

if user_input:
    st.session_state.messages.append(("user", user_input))

    if chat_engine:
        with st.spinner("🤖 Đang suy nghĩ..."):
            response = chat_engine.chat(user_input)
            bot_reply = response.response
            st.session_state.messages.append(("assistant", bot_reply))
    else:
        st.session_state.messages.append(
            ("assistant", "⚠ Vui lòng upload ít nhất 1 PDF."))

    st.rerun()
