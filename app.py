import streamlit as st
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from supabase import create_client

# ========== CONFIGURATION ==========

st.set_page_config(
    page_title="CHAT BOT RAG",
    page_icon="🤖",
    layout="centered"
)

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
            background-color: #f9f9f9;
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

# ========== INITIALISATION ==========

def initialize_rag_components():
    try:
        client = create_client(supabase_url, supabase_key)
        embedding = OpenAIEmbeddings(openai_api_key=openai_key)

        vectorstore = SupabaseVectorStore(
            client=client,
            embedding=embedding,
            table_name="documents",
        )

        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
Tu es un assistant expert. En te basant uniquement sur le CONTEXTE ci-dessous, réponds à la QUESTION. 
Si tu ne sais pas, réponds simplement "Je ne sais pas".

CONTEXTE:
{context}

QUESTION:
{question}

Réponse :
"""
        )

        llm = ChatOpenAI(
            openai_api_key=openai_key,
            model_name="gpt-3.5-turbo",
            temperature=0,
            max_tokens=1500
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
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
        for j, doc in enumerate(sources):
            meta = doc.metadata
            st.markdown(f"""
                <div class='source-document'>
                <b>Source {j+1} - page {meta.get("page", "?")} :</b><br>
                <pre>{doc.page_content}</pre>
                </div>
            """, unsafe_allow_html=True)

# ========== INTERACTION ==========
user_input = st.text_input("❓ Posez votre question ici :")

if user_input:
    with st.spinner("Recherche de la réponse..."):
        response = st.session_state.qa_chain.invoke({"query": user_input})
        answer = response["result"]
        sources = response["source_documents"]

        st.session_state.chat_history.append((user_input, answer, sources))
        display_chat_history()
