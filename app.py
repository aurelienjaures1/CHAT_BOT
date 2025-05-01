import streamlit as st
from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.chains import RetrievalQA
from supabase import create_client

# ========== CONFIGURATION ==========
st.set_page_config(page_title="CHAT BOT", page_icon="🤖")
st.markdown(
    """
    <h1 style='text-align: center; color: #4A90E2;'>🤖 CHAT BOT MÉDICAL 📚</h1>
    <h4 style='text-align: center; color: #555;'>Posez toutes vos questions basées sur vos documents PDF 📄</h4>
    <hr style='border:1px solid #4A90E2'>
    """,
    unsafe_allow_html=True
)

# Charger les variables d'environnement
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

# ========== INIT ========== 
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
    max_tokens=2048,  # Plus de tokens pour de longues réponses
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# ========== APP ==========
question = st.text_input("📝 Entrez votre question ici :", placeholder="Exemple : Que sont les mouvements anormaux liés au sommeil ?")

if question:
    result = qa_chain({"query": question})

    # 🧠 Affichage de la réponse
    st.markdown("### 🧠 Réponse de l'IA :")
    st.success(result["result"])

    # 📚 Affichage des sources utilisées
    if result.get("source_documents"):
        st.markdown("## 📚 Sources utilisées :")
        for i, doc in enumerate(result["source_documents"], 1):
            st.markdown(f"**— Extrait {i} —**")
            st.write(doc.page_content)
            st.caption(f"📄 Source : {doc.metadata.get('source', 'Inconnue')}, Page : {doc.metadata.get('page', '?')}")
            st.markdown("---")
