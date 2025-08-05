# ---------------------------------------------------------
# Explore California - AI Travel App (LangChain 0.2+ Ready)
# ---------------------------------------------------------

import os
import faiss
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from typing import List
import warnings

from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.docstore.document import Document
from langchain_core._api.deprecation import LangChainDeprecationWarning

warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ========== 1. Config & Setup ==========
st.set_page_config(page_title="Explore California AI Travel", layout="wide", initial_sidebar_state="expanded")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

# ========== 2. Env Vars ==========
STREAMLIT_PASSWORD = os.getenv("STREAMLIT_PASSWORD", "linkedin-learning")
OPEN_ROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "all-MiniLM-L6-v2")

# ========== 3. Secure Login ==========
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "just_authenticated" not in st.session_state:
    st.session_state.just_authenticated = False

if not st.session_state.authenticated:
    st.title("🔐 Secure Login")
    password = st.text_input("Enter password", type="password")
    if password == STREAMLIT_PASSWORD:
        st.session_state.authenticated = True
        st.session_state.just_authenticated = True
        st.rerun()
    elif password:
        st.error("Incorrect password")
    st.stop()

if st.session_state.just_authenticated:
    st.session_state.just_authenticated = False
    st.rerun()

# ========== 4. Load Models and Data ==========

@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)

@st.cache_resource
def load_location_vectorstore(_embedder):
    df = pd.read_csv("data/locations.csv")
    docs = [Document(page_content=text, metadata={"name": name}) for name, text in zip(df["location_name"], df["text_data"].fillna(""))]
    vs = FAISS.from_documents(docs, _embedder)
    return df, vs

@st.cache_resource
def load_products():
    return pd.read_csv("data/products.csv")

embedder = load_embedding_model()
df_locations, loc_vectorstore = load_location_vectorstore(embedder)
df_products = load_products()

# ========== 5. Memory + Prompt + Chain ==========
chat_history = StreamlitChatMessageHistory(key="langchain_messages")
memory = ConversationBufferMemory(
    chat_memory=chat_history,
    return_messages=True,
    memory_key="history",
    input_key="query",
    output_key="text"  # ✅ FIXED: align with LLMChain default
)

llm = ChatOpenAI(
    model_name="mistralai/mistral-small-3.2-24b-instruct:free",
    openai_api_key=OPEN_ROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.7
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful travel assistant for Explore California. Use the context and product details provided."),
    ("human", "Context:\n{context}\n\nProducts:\n{products}\n\nQuestion: {query}")
])

rag_chain = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=False)

# ========== 6. Utility Functions ==========

def search_locations(query: str, top_k=3):
    results = loc_vectorstore.similarity_search(query, k=top_k)
    chunks = [f"{doc.metadata['name']}: {doc.page_content}" for doc in results]
    return "\n\n".join(chunks), [doc.metadata["name"] for doc in results]

def match_products(context: str, top_k=3):
    try:
        context_embed = embedder.embed_query(context)
        descs = df_products["description"].fillna("").tolist()
        product_embeds = embedder.embed_documents(descs)
        sims = np.dot(product_embeds, context_embed)
        top_indices = np.argsort(sims)[-top_k:][::-1]
        return df_products.iloc[top_indices]["product_name"].tolist()
    except Exception:
        st.warning("⚠️ Product match fallback")
        return df_products.head(top_k)["product_name"].tolist()

def format_product_descriptions(names: List[str]) -> str:
    blocks = []
    for _, row in df_products[df_products["product_name"].isin(names)].iterrows():
        blocks.append(f"**{row['product_name']}**  \n{row.get('description', '')}  \nDuration: {row.get('duration', 'N/A')} | Price: ${int(row['price_usd']):,}  \nDifficulty: {row.get('difficulty')} | Audience: {row.get('demographics')}")
    return "\n\n".join(blocks)

def get_followups(query, answer):
    prompt = f"Suggest 3 follow-up questions based on this Q&A.\n\nQ: {query}\nA: {answer}\n\nJust list the questions."
    result = llm.invoke([("human", prompt)]).content
    return [q.strip("-• ") for q in result.strip().split("\n") if q.strip()]

# ========== 7. App State ==========
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("followups", [])
st.session_state.setdefault("last_llm_messages", [])
st.session_state.setdefault("trigger_llm", False)
st.session_state.setdefault("current_query", None)

# ========== 8. Streamlit Layout ==========
st.title("🏜️ Explore California - AI Travel Assistant")
st.text("Interact with our AI-powered travel assistant to explore California's best locations and tours!")

with st.sidebar:
    st.subheader("Python LangChain RAG LLM Application")
    st.markdown("This is a RAG LLM app built using the **LangChain** AI framework, running on Google Colab using Ngrok and Mistral from OpenRouter.")
    st.markdown("🔗 [GitHub Repo](https://github.com/LinkedInLearning/applied-AI-and-machine-learning-for-data-practitioners-5932259)")
    if st.button("🔄 Start Over"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if st.session_state.last_llm_messages:
    with st.expander("🧾 View LLM Request Payload"):
        st.json({
            "model": llm.model_name,
            "messages": [
                {"role": "system" if m.__class__.__name__ == "SystemMessage" else "user", "content": m.content}
                for m in st.session_state.last_llm_messages
            ]
        })

# ========== 9. Input Handling ==========
user_input = st.chat_input("Ask me something about California travel...")
if user_input:
    st.session_state.current_query = user_input
    st.session_state.trigger_llm = True
    st.rerun()

if st.session_state.trigger_llm and st.session_state.current_query:
    query = st.session_state.current_query
    with st.spinner("🤖 Thinking..."):
        context, matched_locs = search_locations(query)
        product_names = match_products(context)
        product_block = format_product_descriptions(product_names)

        result = rag_chain.invoke({"query": query, "context": context, "products": product_block})
        response = result["text"]  # ✅ FIXED: use the correct key from LLMChain

        st.session_state.chat_history.append({"role": "user", "content": query})
        st.session_state.chat_history.append({"role": "assistant", "content": response})

        st.session_state.last_llm_messages = memory.chat_memory.messages.copy()
        st.session_state.followups = get_followups(query, response)

        st.session_state.trigger_llm = False
        st.session_state.current_query = None
        st.rerun()

if st.session_state.followups:
    st.markdown("### 🤔 Suggested Follow-Up Questions")
    cols = st.columns(len(st.session_state.followups))
    for i, (col, q) in enumerate(zip(cols, st.session_state.followups)):
        if col.button(q, key=f"fup_{i}"):
            st.session_state.current_query = q
            st.session_state.trigger_llm = True
            st.rerun()
