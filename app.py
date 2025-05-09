import streamlit as st
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.chains import RetrievalQA
from supabase import create_client
import time

# ========== CONFIGURATION ==========
st.set_page_config(
    page_title="CHAT BOT RAG",
    page_icon="🤖",
    layout="centered"
)

# 🎨 CSS custom
st.markdown("""
    <style>
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
        }
        .chat-message.user { background-color: #e6f7ff; }
        .chat-message.assistant { background-color: #f0f0f0; }
        .source-document {
            border-left: 3px solid #4CAF50;
            padding-left: 10px;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 CHAT BOT - Posez vos questions")
st.write("Utilise la puissance de l'IA pour explorer vos documents !")

# ========== SESSION STATE ==========
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "is_initialized" not in st.session_state:
    st.session_state.is_initialized = False

# ========== LECTURE DES SECRETS ==========
openai_key = st.secrets["OPENAI_API_KEY"]
supabase_url = st.secrets["SUPABASE_URL"]
supabase_key = st.secrets["SUPABASE_SERVICE_KEY"]

# ========== INITIALISATION DES COMPOSANTS ==========
def initialize_rag_components():
    try:
        client = create_client(supabase_url, supabase_key)
        embedding = OpenAIEmbeddings(openai_api_key=openai_key)
        vectorstore = SupabaseVectorStore(
            client=client,
            embedding=embedding,
            table_name="documents",
        )
        retriever = vectorstore.as_retriever()
        llm = ChatOpenAI(
            openai_api_key=openai_key,
            model_name="gpt-3.5-turbo",
            temperature=0,
            max_tokens=1500
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )
        st.session_state.qa_chain = qa_chain
        st.session_state.is_initialized = True
    except Exception as e:
        st.error(f"❌ Erreur d'initialisation : {e}")
        st.stop()

if not st.session_state.is_initialized:
    with st.spinner("Initialisation..."):
        initialize_rag_components()

# ========== AFFICHAGE HISTORIQUE ==========
def display_chat_history():
    for i, (question, answer, sources) in enumerate(st.session_state.chat_history):
        st.markdown(
            f"<div class='chat-message user'><b>Vous :</b> {question}</div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div class='chat-message assistant'><b>Assistant :</b> {answer}</div>",
            unsafe_allow_html=True
        )
        if sources:
            with st.expander(f"📚 Sources utilisées (Question {i+1})"):
                for j, doc in enumerate(sources[:5]):
                    page =
