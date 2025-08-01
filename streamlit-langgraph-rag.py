# ---------------------------------------------------------
# Explore California - AI Travel App (LangGraph Version)
# ---------------------------------------------------------

import os
import faiss
import pandas as pd
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from typing import TypedDict, List

from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
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
        model_name="mistralai/mistral-small-3.2-24b-instruct:free",
        openai_api_key=OPEN_ROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.7
    )

    return embedder, location_vs, df_products, llm

embedder, location_vs, df_products, llm = load_models_and_data()

# ========== 6. LangGraph Tools ==========
@tool
def search_locations(query: str) -> dict:
    """Retrieve semantic matches from location descriptions."""
    docs = location_vs.similarity_search(query, k=3)
    context = "\n\n".join([f"{doc.metadata['name']}: {doc.page_content}" for doc in docs])
    return {"query": query, "context": context}

@tool
def match_products(context: str) -> dict:
    """Find relevant tour products from semantic context."""
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
        blocks.append(f"""**{row['product_name']}**  
{row.get('description', 'No description available.')}  
Duration: {duration} | Price: {price}  
Difficulty: {row.get('difficulty', 'Unknown')} | Audience: {row.get('demographics', 'General')}""")
    return {"products": "\n\n".join(blocks)}

@tool
def generate_answer(query: str, context: str, products: str, chat_history: List[dict]) -> dict:
    """Use the LLM to generate a helpful response with context + products."""
    messages = [
        {"role": "system", "content": (
            "You are a helpful travel assistant for the Explore California business. "
            "Use the context and conversation history to assist the user. "
            "Make sure to bold format any product names or location names."
        )}
    ]
    for turn in chat_history[-6:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nRelevant Tour Products:\n{products}\n\nQuestion: {query}"
    })
    result = llm.invoke(messages)
    return {"answer": result.content}

@tool
def suggest_followups(query: str, answer: str) -> dict:
    """Generate 3 follow-up questions based on previous Q&A."""
    prompt = f"""
    Based on the user's question and your response, suggest three follow-up prompts that the user might ask next:

    User Question: {query}
    Assistant Answer: {answer}

    Only return the three follow-up prompts separated by newlines.
    """
    result = llm.invoke([HumanMessage(content=prompt)])
    questions = [q.strip("- •\n ") for q in result.content.strip().split("\n") if q.strip()]
    return {"followups": questions}

# ========== 7. Build LangGraph Workflow ==========
builder = StateGraph(AppState)
builder.add_node("search_locations", search_locations)
builder.add_node("match_products", match_products)
builder.add_node("generate_answer", generate_answer)
builder.add_node("suggest_followups", suggest_followups)
builder.set_entry_point("search_locations")
builder.add_edge("search_locations", "match_products")
builder.add_edge("match_products", "generate_answer")
builder.add_edge("generate_answer", "suggest_followups")
builder.add_edge("suggest_followups", END)
app = builder.compile()

# ========== 8. Streamlit UI ==========
st.title("🏜️ Explore California - AI Travel Assistant")
st.text("Interact with our AI-powered travel assistant to explore California's best locations and tours!")

with st.sidebar:
    st.subheader("Python LangGraph RAG LLM Application")
    st.text("This is a RAG LLM app built using the **LangGraph** AI framework, running on Google Colab using Ngrok and Mistral from OpenRouter.")
    st.markdown("🔗 [GitHub Repo](https://github.com/LinkedInLearning/applied-AI-and-machine-learning-for-data-practitioners-5932259)")
    if st.button("🔄 Start Over"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

query = st.chat_input("Ask something about California travel...")
if query:
    with st.spinner("🧠 Thinking..."):
        chat_history = st.session_state.get("chat_history", [])
        result = app.invoke({
            "query": query,
            "chat_history": chat_history
        })
        st.session_state.stored_context = result["context"]
        st.session_state.stored_products = result["products"]
        st.session_state.last_result = result
        st.session_state.chat_history = chat_history + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": result["answer"]}
        ]
        st.rerun()

if "chat_history" in st.session_state:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

if "last_result" in st.session_state:
    result = st.session_state.last_result
    
    with st.expander("🧾 View LLM Prompt Context", expanded=False):
        st.markdown("**Context**")
        st.markdown(result.get("context", ""))
        st.markdown("**Matched Products**")
        st.markdown(result.get("products", ""))

    if result.get("followups"):
        st.markdown("### 🤔 Suggested Follow-Up Questions")
        cols = st.columns(3)  # Create 3 columns
        for i, q in enumerate(result["followups"]):
            col = cols[i % 3]  # Rotate through the columns
            with col:
                if st.button(q, key=f"fup_{i}"):
                    st.session_state.chat_history += [
                        {"role": "user", "content": result["query"]},
                        {"role": "assistant", "content": result["answer"]}
                    ]
                    with st.spinner("🔁 Thinking..."):
                        new_result = generate_answer.invoke({
                            "query": q,
                            "context": st.session_state.stored_context,
                            "products": st.session_state.stored_products,
                            "chat_history": st.session_state.chat_history
                        })
                        followups = suggest_followups.invoke({
                            "query": q,
                            "answer": new_result["answer"]
                        })
                        combined = {
                            "query": q,
                            "context": st.session_state.stored_context,
                            "products": st.session_state.stored_products,
                            "answer": new_result["answer"],
                            "followups": followups["followups"]
                        }
                        st.session_state.last_result = combined
                        st.session_state.chat_history += [
                            {"role": "user", "content": q},
                            {"role": "assistant", "content": new_result["answer"]}
                        ]
                        st.rerun()
