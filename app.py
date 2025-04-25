import streamlit as st
from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.chains import RetrievalQA
from supabase import create_client

# ========== 🎨 Style Pro ========== #
st.set_page_config(page_title="CHAT BOT", page_icon="🤖")

st.markdown("""
    <style>
    body {
        background-color: #f0f2f6;
    }
    .stTextInput > div > div > input {
        font-size: 18px;
        padding: 12px;
        border: 2px solid #4CAF50;
        border-radius: 10px;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 10px 24px;
        border: none;
        border-radius: 8px;
    }
    .stMarkdown {
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("# 🤖 CHAT BOT")
st.markdown("### 🔍 Posez toutes vos questions à vos documents PDF directement ici 📚")

# ========== 🔐 Configuration API ========== #
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

# ========== ⚡ Initialisation Backend ========== #
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
    temperature=0,
    model_name="gpt-3.5-turbo",
    max_tokens=1024
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# ========== 🖋 Interface utilisateur ========== #
question = st.text_input("📝 Votre question :", "")

if question:
    with st.spinner('✏️ Génération de la réponse...'):
        result = qa_chain({"query": question})

    st.markdown("### 🧠 Réponse :")
    st.success(result["result"])

    # Affichage des sources
    if result.get("source_documents"):
        st.markdown("### 📚 Sources :")
        for doc in result["source_documents"]:
            source = f"Page {doc.metadata.get('page', '?')} - {doc.metadata.get('source', 'Inconnue')}"
            st.markdown(f"- {source}")
