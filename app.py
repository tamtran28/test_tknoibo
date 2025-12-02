import streamlit as st
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface.base import HuggingFaceEmbedding

# ============================
# PAGE CONFIG
# ============================
st.set_page_config(page_title="RAG PDF Chatbot", page_icon="🤖", layout="wide")

st.title("🤖 RAG Chatbot PDF – Streamlit Cloud + Groq API (FREE)")

# ============================
# FILE UPLOAD
# ============================
st.sidebar.header("📄 Upload PDF")
uploaded_files = st.sidebar.file_uploader(
    "Tải lên file PDF",
    type=["pdf"],
    accept_multiple_files=True
)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# LƯU FILE PDF LÊN SERVER
if uploaded_files:
    for file in uploaded_files:
        filepath = os.path.join(DATA_DIR, file.name)
        with open(filepath, "wb") as f:
            f.write(file.getbuffer())
    st.sidebar.success("✔ PDF đã được lưu!")

# ============================
# BUILD INDEX
# ============================
@st.cache_resource
def load_index():
    pdfs = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    if not pdfs:
        return None

    docs = SimpleDirectoryReader(DATA_DIR).load_data()

    # LLM miễn phí Groq
    llm = Groq(
        model="llama3-8b-8192",
        api_key=st.secrets["GROQ_API_KEY"]
    )

    # Embedding miễn phí HuggingFace
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
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
    chat_engine = index.as_chat_engine(chat_mode="condense_question")
else:
    chat_engine = None

# ============================
# CHAT UI
# ============================
st.subheader("💬 Chat với PDF")

if "messages" not in st.session_state:
    st.session_state.messages = []

# HIỂN THỊ CHAT CŨ
for role, msg in st.session_state.messages:
    st.chat_message(role).markdown(msg)

# INPUT
user_msg = st.chat_input("Nhập câu hỏi...")

if user_msg:
    st.session_state.messages.append(("user", user_msg))

    if chat_engine:
        with st.spinner("Đang xử lý..."):
            reply = chat_engine.chat(user_msg)
            bot_reply = reply.response
            st.session_state.messages.append(("assistant", bot_reply))
    else:
        st.session_state.messages.append(("assistant", "⚠ Hãy upload ít nhất 1 PDF!"))

    st.rerun()
