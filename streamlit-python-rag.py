# ---------------------------------------------------------
# Explore California - AI Travel App (RAG-powered)
# ---------------------------------------------------------
# This app answers travel questions using:
# - Local semantic search (SentenceTransformer + FAISS)
# - Cloud-based LLM reasoning (OpenRouter API)
# - Tour and location datasets
# ---------------------------------------------------------

# ========== 1. Library Imports & Configuration ==========
import os
import faiss
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from typing import List
import warnings

# Streamlit page configuration
st.set_page_config(
    page_title="Explore California AI Travel",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Suppress future warnings for cleaner logs
warnings.filterwarnings("ignore", category=FutureWarning)

# Avoid tokenizer parallelism errors
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables from .env file
load_dotenv()

# Load GPU layer configuration for quantized models (if applicable)
try:
    GPU_LAYERS = int(os.getenv("GPU_LAYERS", 16))
except ValueError:
    GPU_LAYERS = 16
    st.warning("⚠️ Invalid GPU_LAYERS value in .env — defaulting to 16")

# Load API keys and authentication credentials
STREAMLIT_PASSWORD = os.getenv("STREAMLIT_PASSWORD", "linkedin-learning")
OPEN_ROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")


# ========== 2. Secure Login ==========
# Basic password gate to restrict app access
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "just_authenticated" not in st.session_state:
    st.session_state.just_authenticated = False

# Prompt user for password if not already authenticated
if not st.session_state.authenticated:
    st.title("🔐 Secure Login")
    password = st.text_input("Enter password", type="password")
    if password == STREAMLIT_PASSWORD:
        st.session_state.authenticated = True
        st.session_state.just_authenticated = True
        st.rerun()  # Restart app flow after login
    elif password:
        st.error("Incorrect password")
    st.stop()  # Prevent rest of app from rendering

# Reset flag and rerun after initial login
if st.session_state.just_authenticated:
    st.session_state.just_authenticated = False
    st.rerun()


# ========== 3. Load Models ==========
# Load the sentence transformer embedding model from local path
@st.cache_resource
def load_local_embedding_model():
    return SentenceTransformer(os.getenv("EMBEDDING_MODEL_PATH"))

# Create OpenAI-compatible client using OpenRouter
@st.cache_resource
def get_openrouter_client():
    return OpenAI(api_key=OPEN_ROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")

def get_response_and_followups(
    query: str,
    context: str,
    product_context: str,
    chat_history: List[dict],
    debug: bool = False
) -> dict:
    """
    Calls the OpenRouter LLM to generate a structured answer and follow-up questions in a single response.

    Args:
        query: User's input question.
        context: Retrieved location context.
        product_context: Matched tour products, formatted for display.
        chat_history: Previous user/assistant turns.
        debug: Whether to print/log the full response.

    Returns:
        A dictionary with:
            - "answer": main assistant response (cleaned)
            - "followups": list of 3 suggested follow-up questions
            - "total_input_tokens": input token count (if available)
            - "total_completion_tokens": completion token count (if available)
            - "total_tokens": total token count (if available)
    """
    # Prepare system and conversation messages
    system_msg = {
        "role": "system",
        "content": (
            "You are a helpful travel assistant for Explore California.\n"
            "Use the provided context and tour product suggestions to answer the user's question.\n"
            "Always format any tour product or location names in **bold**.\n"
            "At the end of your answer, suggest 3 relevant follow-up questions.\n"
            "Format your response like this:\n\n"
            "**Answer:** <your full assistant reply>\n\n"
            "**Suggested Follow-Ups:**\n"
            "- Question 1\n"
            "- Question 2\n"
            "- Question 3"
        )
    }

    messages = [system_msg]

    # Add trimmed chat history (last 6 turns)
    for turn in chat_history[-6:]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    # Construct user message with full context
    user_msg = {
        "role": "user",
        "content": (
            f"Context:\n{context}\n\n"
            f"Relevant Tour Products:\n{product_context}\n\n"
            f"Question: {query}"
        )
    }
    messages.append(user_msg)

    # Send request to OpenRouter
    client = get_openrouter_client()
    response = client.chat.completions.create(
        model="mistralai/mistral-small-3.2-24b-instruct:free",
        messages=messages,
        max_tokens=1400,
        temperature=0.7
    )

    # Parse structured output
    content = response.choices[0].message.content.strip()
    answer = ""
    followups = []

    if "**Suggested Follow-Ups:**" in content:
        answer_part, followup_part = content.split("**Suggested Follow-Ups:**", 1)
        answer = answer_part.replace("**Answer:**", "").strip()
        followups = [
            line.strip("-• \n") for line in followup_part.strip().splitlines()
            if line.strip()
        ]
    else:
        answer = content
        
    return {
        "answer": answer,
        "followups": followups[:3]
    }

# ========== 4. Load and Index Data ==========
@st.cache_resource
def load_datasets_and_faiss_indexes(_embed_model):
    # Load location dataset and encode it into vector space
    df_locations = pd.read_csv("data/locations.csv")
    location_texts = df_locations["text_data"].fillna("").tolist()
    location_embeddings = _embed_model.encode(location_texts, show_progress_bar=True)

    # Build FAISS index for semantic similarity search over locations
    index_locations = faiss.IndexFlatL2(location_embeddings[0].shape[0])
    index_locations.add(location_embeddings)

    # Load product metadata
    df_products = pd.read_csv("data/products.csv")
    return df_locations, index_locations, df_products

# Retrieve most relevant location descriptions given a user query
def retrieve_location_context(query, embed_model, loc_index, df_loc, top_k=3):
    query_vec = embed_model.encode([query])
    D_loc, I_loc = loc_index.search(query_vec, top_k)
    loc_chunks = [f"{df_loc.iloc[i]['location_name']}: {df_loc.iloc[i]['text_data']}" for i in I_loc[0]]
    matched_locations = [df_loc.iloc[i]['location_name'] for i in I_loc[0]]
    return "\n\n".join(loc_chunks), matched_locations

# Find relevant products based on the context (semantic similarity)
def get_relevant_products_semantically(context, df_products, embed_model, top_k=3):
    try:
        context_embedding = embed_model.encode([context])
        product_embeddings = embed_model.encode(df_products["description"].fillna("").tolist())
        similarities = cosine_similarity(context_embedding, product_embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        product_names = df_products.iloc[top_indices]["product_name"].tolist()
        if not product_names:
            product_names = df_products.head(top_k)["product_name"].tolist()
    except Exception:
        st.warning("⚠️ Semantic product retrieval failed — using default products.")
        product_names = df_products.head(top_k)["product_name"].tolist()
    return product_names

# Format product metadata into human-readable Markdown blocks
def format_product_context(product_names, df_products):
    rows = df_products[df_products["product_name"].isin(product_names)]
    product_blocks = []
    for _, row in rows.iterrows():
        block = f"""**{row['product_name']}**  
{row.get('description', 'No description available.')}  
Duration: {row.get('duration', 'Duration not specified')} | Price: ${int(row['price_usd']):,}  
Difficulty: {row.get('difficulty', 'Unknown')}. Target: {row.get('demographics', 'General audience')}"""
        product_blocks.append(block)
    return "\n\n".join(product_blocks)


# ========== 5. UI State & Header ==========
# App title and description
st.title("🏜️ Explore California - AI Travel")
st.text("Interact with our AI-powered travel assistant to explore California's best locations and tours!")

# Sidebar info and reset button
with st.sidebar:
    st.subheader("Pure Python RAG LLM Application")
    st.text("This is a RAG app built using pure Python, running on Google Colab using Ngrok and Mistral from OpenRouter.")
    st.markdown("🔗 **[GitHub Source Code](https://github.com/LinkedInLearning/applied-AI-and-machine-learning-for-data-practitioners-5932259/blob/main/streamlit-python-rag.py)**")
    if st.button("🔄 Start New Chat"):
        for key in ["qa_history", "chat_history", "followup_suggestions", "trigger_llm", "current_query", "context_retrieved", "initial_context", "initial_products"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# Initialize session state for chat flow
for key in ["qa_history", "chat_history", "followup_suggestions"]:
    if key not in st.session_state:
        st.session_state[key] = []
if "trigger_llm" not in st.session_state:
    st.session_state.trigger_llm = False
if "current_query" not in st.session_state:
    st.session_state.current_query = None
if "context_retrieved" not in st.session_state:
    st.session_state.context_retrieved = False


# ========== 6. Load Data + Chat Interface ==========
# Load embedding model and datasets
local_embedding_model = load_local_embedding_model()
df_locations, faiss_loc_index, df_products = load_datasets_and_faiss_indexes(local_embedding_model)

# Render previous chat history with context expansion
for idx, turn in enumerate(st.session_state.chat_history):
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn["role"] == "assistant":
            qa_idx = idx // 2
            if qa_idx < len(st.session_state.qa_history):
                qa = st.session_state.qa_history[qa_idx]
                context = qa.get("context", "")
                product_context = format_product_context(qa.get("products", []), df_products)
                query = qa.get("query", "")
                messages = [
                    {"role": "system", "content": "You are a helpful travel assistant for the Explore California business. Use the context and conversation history to assist the user. Make sure to bold format any product names or location names."}
                ]
                prior = st.session_state.chat_history[max(0, idx - 6):idx]
                for h in prior:
                    messages.append({"role": h["role"], "content": h["content"]})
                messages.append({"role": "user", "content": f"Context:\n{context}\n\nRelevant Tour Products:\n{product_context}\n\nQuestion: {query}"})
                with st.expander("🧾 View Full LLM Request Payload", expanded=False):
                    st.json({
                        "model": "mistralai/mistral-small-3.2-24b-instruct:free",
                        "messages": messages,
                        "max_tokens": 1028,
                        "temperature": 0.7
                    })

# User input at bottom of screen (chat-style)
user_input = st.chat_input("Ask me something about California travel...")
if user_input:
    st.session_state.current_query = user_input
    st.session_state.trigger_llm = True
    st.rerun()


# ========== 7. Trigger RAG + LLM ==========
if st.session_state.trigger_llm and st.session_state.current_query:
    query = st.session_state.current_query
    with st.spinner("🔍 Thinking..."):
        # Retrieve semantic context and product matches only once per query
        if not st.session_state.context_retrieved:
            context, matched_locations = retrieve_location_context(query, local_embedding_model, faiss_loc_index, df_locations)
            relevant_products = get_relevant_products_semantically(context, df_products, local_embedding_model)
            product_context = format_product_context(relevant_products, df_products)
            st.session_state.initial_context = context
            st.session_state.initial_products = product_context
            st.session_state.context_retrieved = True
        else:
            context = st.session_state.initial_context
            product_context = st.session_state.initial_products
            relevant_products = []

        # 🧠 Call LLM once to get both answer and follow-ups
        result = get_response_and_followups(
            query=query,
            context=context,
            product_context=product_context,
            chat_history=st.session_state.chat_history
        )
        answer = result["answer"]
        followups = result["followups"]

        # 📜 Log to session state
        st.session_state.chat_history.append({"role": "user", "content": query})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.session_state.qa_history.append({
            "query": query,
            "context": context,
            "products": relevant_products,
            "cloud_response": answer
        })
        st.session_state.followup_suggestions = followups
        st.session_state.trigger_llm = False
        st.session_state.current_query = None
        st.rerun()


# ========== 8. Follow-Up Suggestions ==========
# Display clickable follow-up suggestions from LLM
if st.session_state.followup_suggestions:
    st.markdown("### 🤔 Suggested Follow-Up Questions")
    cols = st.columns(len(st.session_state.followup_suggestions))
    for i, (col, question) in enumerate(zip(cols, st.session_state.followup_suggestions)):
        if col.button(question, key=f"followup_{i}"):
            st.session_state.current_query = question
            st.session_state.trigger_llm = True
            st.rerun()
