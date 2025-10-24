# ---------------------------------------------------------
# Explore California - AI Travel App (LangGraph Conditional RAG)
# ---------------------------------------------------------

import os
import pandas as pd
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from typing import TypedDict, List

from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS as LangGraphFAISS

# ========== 1. Configuration ==========
st.set_page_config(page_title="Explore California - LangGraph", layout="wide")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

# ========== 2. Environment Variables ==========
STREAMLIT_PASSWORD = os.getenv("STREAMLIT_PASSWORD", "linkedin-learning")
OPEN_ROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "all-MiniLM-L6-v2")

# ========== 3. Streamlit Authentication ==========
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

# ========== 4. LangGraph App State ==========
class AppState(TypedDict):
    query: str
    context: str
    products: str
    answer: str
    followups: List[str]
    chat_history: List[dict]
    followup_mode: bool
    start_node: str

# ========== 5. Load Models & Data ==========
@st.cache_resource
def load_models_and_data():
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)

    df_locations = pd.read_csv("data/locations.csv")
    docs = [Document(page_content=row["text_data"], metadata={"name": row["location_name"]})
            for _, row in df_locations.iterrows()]
    location_vs = LangGraphFAISS.from_documents(docs, embedder)

    df_products = pd.read_csv("data/products.csv")

    llm = ChatOpenAI(
        model_name="mistralai/mistral-7b-instruct:free",
        openai_api_key=OPEN_ROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.7
    )

    return embedder, location_vs, df_products, llm

embedder, location_vs, df_products, llm = load_models_and_data()

# ========== 6. LangGraph Tools ==========
@tool
def search_locations(query: str) -> dict:
    """Search for semantically similar locations from the vector store using the user's query."""
    docs = location_vs.similarity_search(query, k=3)
    context = "\n\n".join([f"{doc.metadata['name']}: {doc.page_content}" for doc in docs])
    return {"context": context}

@tool
def match_products(context: str) -> dict:
    """Match relevant travel products using the provided context via vector similarity."""
    try:
        context_embed = embedder.embed_query(context)
        descs = df_products["description"].fillna("").tolist()
        product_embeds = embedder.embed_documents(descs)
        sims = np.dot(product_embeds, context_embed)
        top_k = 3
        top_indices = np.argsort(sims)[-top_k:][::-1]
    except Exception:
        st.warning("⚠️ Semantic matching failed — defaulting to top products.")
        top_indices = df_products.head(3).index.tolist()

    blocks = []
    for i in top_indices:
        row = df_products.iloc[i]
        duration = row.get("duration_days", "Duration not specified")
        if pd.isna(duration):
            duration = "Duration not specified"
        price = f"${int(row.get('price_usd', 0)):,}"
        blocks.append(f"""**{row['product_name']}**  \
{row.get('description', 'No description available.')}  \
Duration: {duration} | Price: {price}  \
Difficulty: {row.get('difficulty', 'Unknown')} | Audience: {row.get('demographics', 'General')}""")
    return {"products": "\n\n".join(blocks)}

@tool
def generate_answer_and_followups(query: str, context: str, products: str, chat_history: List[dict]) -> dict:
    """
    Generate an assistant response and 3 follow-up questions using the LLM in one call.
    """
    messages = [
        {"role": "system", "content": (
            "You are a helpful travel assistant for Explore California.\n"
            "Use the provided context and product information to answer the user's question.\n"
            "If relevant tour products are mentioned, format them in **bold**.\n"
            "After the answer, suggest 3 concise follow-up questions the user might ask next.\n"
            "Format your response like this:\n\n"
            "**Answer:** <your assistant reply>\n\n"
            "**Suggested Follow-Ups:**\n"
            "- Question 1\n"
            "- Question 2\n"
            "- Question 3"
        )}
    ]

    for turn in chat_history[-6:]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    user_msg = {
        "role": "user",
        "content": (
            f"Context:\n{context}\n\n"
            f"Relevant Tour Products:\n{products}\n\n"
            f"Question: {query}"
        )
    }
    messages.append(user_msg)

    response = llm.invoke(messages)
    content = response.content.strip()

    if "**Suggested Follow-Ups:**" in content:
        answer_part, followup_part = content.split("**Suggested Follow-Ups:**", 1)
        answer = answer_part.replace("**Answer:**", "").strip()
        followups = [q.strip("-• \n") for q in followup_part.strip().splitlines() if q.strip()]
    else:
        answer = content
        followups = []

    return {
        "answer": answer,
        "followups": followups[:3]
    }

@tool
def entry_selector(followup_mode: bool) -> dict:
    """Determine whether to begin at search_locations (first query) or generate_answer (follow-up)."""
    return {"start_node": "generate_answer" if followup_mode else "search_locations"}

# ========== 7. Build LangGraph Workflow ==========
builder = StateGraph(AppState)

builder.add_node("entry_selector", entry_selector)
builder.add_node("search_locations", search_locations)
builder.add_node("match_products", match_products)
builder.add_node("generate_answer", generate_answer_and_followups)

builder.set_entry_point("entry_selector")
builder.add_conditional_edges("entry_selector", lambda state: state["start_node"])
builder.add_edge("search_locations", "match_products")
builder.add_edge("match_products", "generate_answer")
builder.add_edge("generate_answer", END)

app = builder.compile()

# ========== 8. Streamlit UI ==========
st.title("🏜️ Explore California - AI Travel Assistant")
st.text("Interact with our AI-powered travel assistant to explore California's best locations and tours!")

with st.sidebar:
    st.subheader("⚙️ Settings")
    st.markdown("This app uses LangGraph + RAG with a Mistral LLM via OpenRouter.")
    st.markdown("🔗 [GitHub Repo](https://github.com/LinkedInLearning/applied-AI-and-machine-learning-for-data-practitioners-5932259/blob/main/streamlit-langgraph-rag.py)")
    if st.button("🔄 Start Over"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
            # Don't reset authentication state
            st.session_state.authenticated = True
            st.session_state.just_authenticated = True
        st.rerun()

# Input from user
query = st.chat_input("Ask me something about California travel...")

# Handle follow-up clicks
followup_query = st.session_state.pop("selected_followup", None)
actual_query = query or followup_query

if actual_query:
    with st.spinner("🧠 Thinking..."):
        chat_history = st.session_state.get("chat_history", [])

        # Use followup_mode if we already have stored context
        is_followup = "stored_context" in st.session_state

        result = app.invoke({
            "query": actual_query,
            "chat_history": chat_history,
            "context": st.session_state.get("stored_context", ""),
            "products": st.session_state.get("stored_products", ""),
            "answer": "",
            "followups": [],
            "followup_mode": is_followup
        })

        # Store context and results
        st.session_state.stored_context = result["context"]
        st.session_state.stored_products = result["products"]
        st.session_state.last_result = result

        # Append to chat history
        st.session_state.chat_history = chat_history + [
            {"role": "user", "content": actual_query},
            {"role": "assistant", "content": result["answer"]}
        ]

        st.rerun()

# Display chat history
if "chat_history" in st.session_state:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# Show context and follow-ups
if "last_result" in st.session_state:
    result = st.session_state.last_result

    with st.expander("🧾 View Context & Products"):
        if "stored_context" in st.session_state:
            st.markdown("**📊 Using previously retrieved context for follow-up question!**")

        st.markdown("**Context:**")
        st.markdown(result.get("context", ""))
        st.markdown("**Matched Products:**")
        st.markdown(result.get("products", ""))

    if result.get("followups"):
        st.markdown("### 🤔 Suggested Follow-Up Questions")
        cols = st.columns(3)
        for i, q in enumerate(result["followups"]):
            col = cols[i % 3]
            with col:
                if st.button(q, key=f"fup_{i}"):
                    with st.spinner("🧠 Thinking..."):
                        st.session_state.selected_followup = q
                        st.rerun()
