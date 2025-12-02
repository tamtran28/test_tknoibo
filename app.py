import streamlit as st
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.groq import Groq
from llama_index.embeddings.openai import OpenAIEmbedding

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

# UI
st.title("🤖 RAG Chatbot - PDF - Miễn phí - Streamlit Cloud")
st.sidebar.header("📄 Upload PDF")

uploaded_files = st.sidebar.file_uploader(
    "Upload một hoặc nhiều PDF",
    type=["pdf"],
    accept_multiple_files=True
)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Lưu PDF
if uploaded_files:
    for file in uploaded_files:
        with open(os.path.join(DATA_DIR, file.name), "wb") as f:
            f.write(file.read())
    st.sidebar.success("✔ PDF uploaded!")

# Lấy API KEY từ secrets
GROQ_KEY = st.secrets["GROQ_API_KEY"]

# Tạo index
@st.cache_resource
def load_index():
    files = os.listdir(DATA_DIR)
    if not files:
        return None

    docs = SimpleDirectoryReader(DATA_DIR).load_data()

    llm = Groq(model="llama3-8b-8192", api_key=GROQ_KEY)
    embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    index = VectorStoreIndex.from_documents(docs, llm=llm, embed_model=embed_model)
    return index

index = load_index()

# Chatbot
if index:
    chat_engine = index.as_chat_engine(chat_mode="condense_question")
else:
    chat_engine = None

st.subheader("💬 Chat với PDF")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiện chat cũ
for role, text in st.session_state.messages:
    st.chat_message(role).markdown(text)

# Input
user_input = st.chat_input("Nhập câu hỏi...")

if user_input:
    st.session_state.messages.append(("user", user_input))
    
    if chat_engine:
        with st.spinner("Đang xử lý..."):
            response = chat_engine.chat(user_input)
            bot_answer = response.response
            st.session_state.messages.append(("assistant", bot_answer))
    else:
        st.session_state.messages.append(("assistant", "❗ Hãy upload PDF trước!"))

    st.experimental_rerun()
