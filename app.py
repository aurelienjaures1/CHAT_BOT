import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.chains import RetrievalQA
from supabase import create_client

# ========== CONFIGURATION ==========

st.set_page_config(
    page_title="CHAT BOT 📚🧠",
    page_icon="🤖",
    layout="centered"
)

# 🎨 Custom CSS
st.markdown("""
    <style>
        body {
            background-color: #f7f9fc;
        }
        .stTextInput input {
            font-size: 18px;
            padding: 0.5rem;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 10px;
            font-size: 18px;
        }
        .stMarkdown {
            font-size: 18px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 CHAT BOT - Posez vos questions à vos PDF 📚")
st.write("Utilise la puissance de l'IA pour explorer vos documents !")

# ========== CHARGEMENT DES CLES .ENV ==========
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

# ========== INITIALISATION ==========
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
    max_tokens=1500  # 🔥 Plus de détails dans les réponses
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# ========== INTERFACE UTILISATEUR ==========
question = st.text_input("✍️ Entrez votre question ici :", placeholder="Exemple : Quels sont les symptômes du SAOS ?")

if st.button("🔍 Lancer la recherche"):
    if question:
        try:
            result = qa_chain({"query": question})
            st.markdown("## 🧠 Réponse de l'IA :")
            st.success(result["result"])

            # 📚 Sources
            if result.get("source_documents"):
                st.markdown("## 📖 Sources utilisées :")
                for doc in result["source_documents"]:
                    page = doc.metadata.get('page', '?')
                    source = doc.metadata.get('source', 'Document inconnu')
                    st.markdown(f"- **Page {page}** : {source}")

        except Exception as e:
            st.error(f"❌ Une erreur est survenue : {e}")
    else:
        st.warning("⚠️ Veuillez entrer une question avant de lancer la recherche.")
