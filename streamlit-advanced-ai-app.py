# ---------------------------------------------------------
# Explore California - AI Travel App (Advanced LangGraph RAG)
# ---------------------------------------------------------

import json
import os
import pandas as pd
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from typing import TypedDict, List, Optional
import joblib

from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS as LangGraphFAISS

# ========== 1. Configuration ==========
st.set_page_config(page_title="Explore California AI", layout="wide")
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
    total_input_tokens: Optional[int]
    total_completion_tokens: Optional[int]
    total_tokens: Optional[int]
    query: str
    context: str
    products: str
    answer: str
    followups: List[str]
    chat_history: List[dict]
    followup_flag: bool
    predicted_product: str
    attributes: List[str]
    topic_label: str

# ========== 5. Load Models & Data ==========
@st.cache_resource
def load_models_and_data():
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)
    
    if not os.path.exists("data/logistic_model_outputs.pkl"):
        st.error("Missing model artifacts. Please add 'logistic_model_outputs.pkl' to the data folder.")
        st.stop()

    model_artefacts = joblib.load("data/logistic_model_outputs.pkl")
    ml_model = model_artefacts["model"]
    ml_label_encoder = model_artefacts["label_encoder"]
    ml_attribute_names = model_artefacts["attribute_names"]

    attr_docs = [Document(page_content=attr, metadata={"name": attr})
                 for attr in ml_attribute_names]
    attr_vs = LangGraphFAISS.from_documents(attr_docs, embedder)

    df_locations = pd.read_csv("data/locations.csv")
    docs = [Document(page_content=row["text_data"], metadata={"name": row["location_name"]})
            for _, row in df_locations.iterrows()]
    location_vs = LangGraphFAISS.from_documents(docs, embedder)

    df_products = pd.read_csv("data/products.csv")

    llm = ChatOpenAI(
        model_name="mistralai/mistral-7b-instruct:free",
        openai_api_key=OPEN_ROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.7,
        verbose=True
    )

    return embedder, ml_model, ml_label_encoder, ml_attribute_names, location_vs, df_products, attr_vs, llm, df_locations

embedder, ml_model, ml_label_encoder, ml_attribute_names, location_vs, df_products, attr_vs, llm, df_locations = load_models_and_data()

# ========== 6. LangGraph Tools ==========
@tool
def find_similar_attributes(query: str) -> dict:
    """
    Use FAISS to find the most semantically relevant attributes for a user query.
    
    Args:
        query: User input describing preferences (e.g., "I love hiking and wine tasting").
    
    Returns:
        A dictionary with a list of selected attributes.
    """
    try:
        matches = attr_vs.similarity_search(query, k=10)
        selected = list({doc.page_content for doc in matches})
        return {"attributes": selected}
    except Exception as _:
        return {"attributes": []}
    
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
def generate_answer_with_followups(
    query: str,
    context: str,
    products: str,
    chat_history: List[dict]
) -> dict:
    """
    Generates an LLM answer to the user's question and suggests 3 follow-up questions in a single call.

    Args:
        query: User's input question.
        context: Retrieved location or attribute context.
        products: Recommended travel products.
        chat_history: Previous user/assistant turns.

    Returns:
        dict with:
            - "answer": main LLM response
            - "followups": list of 3 suggested follow-up questions
            - "total_input_tokens": number of tokens in the prompt
            - "total_completion_tokens": number of tokens in the completion
            - "total_tokens": total tokens used
    """
    # Convert messages to LangChain Message objects
    messages = [
        SystemMessage(content=(
            "You are a helpful travel assistant for Explore California. "
            "Answer the user's travel question using the provided context and products. "
            "If relevant tour products are provided, make sure to include them in your answer and format them in **bold** style. "
            "Then, at the end, suggest 3 concise follow-up questions that the user might ask next. "
            "Respond in the following format:\n\n"
            "**Answer:** <your full assistant response>\n\n"
            "**Suggested Follow-Ups:**\n"
            "- Question 1\n"
            "- Question 2\n"
            "- Question 3"
        ))
    ]

    for turn in chat_history[-6:]:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        elif turn["role"] == "assistant":
            messages.append(AIMessage(content=turn["content"]))

    user_input = (
        f"Context:\n{context}\n\n"
        f"Relevant Tour Products:\n{products}\n\n"
        f"Question: {query}"
    )
    messages.append(HumanMessage(content=user_input))

    # Use generate to capture token usage
    res = llm.generate([messages])  # List of list of messages
    message = res.generations[0][0].message
    usage = res.llm_output.get("token_usage", {})

    output = message.content.strip()
    answer = ""
    followups = []

    if "**Suggested Follow-Ups:**" in output:
        answer_part, followup_part = output.split("**Suggested Follow-Ups:**", 1)
        answer = answer_part.replace("**Answer:**", "").strip()
        followups = [
            line.strip("-• \n") for line in followup_part.strip().splitlines()
            if line.strip()
        ]
    else:
        answer = output

    return {
        "answer": answer,
        "followups": followups[:3],
        "total_input_tokens": usage.get("prompt_tokens"),
        "total_completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


@tool
def predict_product_from_attributes(attributes: List[str]) -> dict:
    """Predict the most likely product using the ML model and selected user attributes."""
    vec = [1 if attr in attributes else 0 for attr in ml_attribute_names]
    df_input = pd.DataFrame([vec], columns=ml_attribute_names)
    pred_index = ml_model.predict(df_input)[0]
    pred_product = ml_label_encoder.inverse_transform([pred_index])[0]
    matched = df_products[df_products["product_name"] == pred_product]
    
    row = matched.iloc[0]
    duration = row.get("duration_days", "Not specified")
    price = f"${int(row.get('price_usd', 0)):,}"
    description = row.get("description", "No description available.")
    difficulty = row.get("difficulty", "Unknown")
    audience = row.get("demographics", "General")
    product_info = f"""**{pred_product}**  
{description}  
Duration: {duration} | Price: {price}  
Difficulty: {difficulty} | Audience: {audience}"""
    return {"predicted_product": pred_product, "products": product_info}

@tool
def get_locations_for_product(predicted_product: str) -> dict:
    """
    Retrieves relevant location descriptions for a predicted product.

    Args:
        predicted_product: The name of the tour product.

    Returns:
        A dictionary with a 'context' string describing matching locations.
    """
    # Match based on product location names
    # First get the locations from the product
    locations_str = df_products[df_products["product_name"] == predicted_product]["locations"].values[0]

    # Split and clean
    location_names = [loc.strip() for loc in locations_str.split(",") if loc.strip()]

    # Match to df_locations
    matched_rows = df_locations[df_locations["location_name"].isin(location_names)]

    context = "\n\n".join([f"{row['location_name']}: {row['text_data']}" for _, row in matched_rows.iterrows()])

    return {"context": context}

@tool
def entry_selector(query: str, followup_flag: bool = False, chat_history: List[dict] = []) -> dict:
    """
    Route the flow and generate a short chat topic label using a single LLM call.

    Args:
        query: The user's input
        followup_flag: Whether this is a follow-up query
        chat_history: Full chat history (for context in topic labeling)

    Returns:
        dict with:
            - "start_node": either 'find_attributes', 'search_locations', or 'generate_answer'
            - "topic_label": a short 5-10 word summary
    """
    if followup_flag:
        return {"start_node": "generate_answer"}

    # Build recent user message history
    user_turns = [m["content"] for m in chat_history if m["role"] == "user"]
    user_turns.append(query)
    chat_history_str = "\n".join(user_turns[-6:])

    prompt = f"""
You are a smart travel assistant.

Step 1: ROUTING
Decide the type of request from the user.
- If they are listing interests or preferences (e.g., "I love hiking and wine tasting"), return: find_attributes
- If they are asking a travel question (e.g., "What are the best places to visit in California?"), return: search_locations
- Otherwise, return: generate_answer

Step 2: TOPIC LABEL
Also generate a short 5-10 word topic label that summarizes the user's interest.
- Be sure to keep articles, prepositions, and conjunctions lowercase unless they are the first word (e.g., "the", "in", "of", "and")
- Capitalize major words (nouns, verbs, adjectives)

ONLY return the result in this format (no extra words):

ROUTE: <one of: find_attributes, search_locations, generate_answer>
TOPIC: <short topic label>

---

Chat History:
{chat_history_str}
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    text = response.content.strip()
    
    # Default route and label if there is hallucination
    label = "New Chat"

    for line in text.splitlines():
        if line.lower().startswith("route:"):
            route = line.split(":", 1)[1].strip().lower()
        elif line.lower().startswith("topic:"):
            label = line.split(":", 1)[1].strip()

    # fallback if parsing fails
    if route not in {"find_attributes", "search_locations"}:
        route = "generate_answer"

    return {
        "start_node": route,
        "topic_label": label or "New Chat"
    }

# ========== 7. Build LangGraph Workflow ==========
builder = StateGraph(AppState)

# Nodes
builder.add_node("entry_selector", entry_selector)
builder.add_node("find_attributes", find_similar_attributes)
builder.add_node("predict_product", predict_product_from_attributes)
builder.add_node("get_product_locations", get_locations_for_product)
builder.add_node("search_locations", search_locations)
builder.add_node("match_products", match_products)
builder.add_node("generate_answer", generate_answer_with_followups)

# Entry point
builder.set_entry_point("entry_selector")

# --- Routing from entry_selector based on LLM decision ---
def determine_route(state):
    return state.get("start_node", "search_locations") 

builder.add_conditional_edges("entry_selector", determine_route)

# Regular RAG flow
builder.add_edge("search_locations", "match_products")
builder.add_edge("match_products", "generate_answer")

# Attribute-based prediction flow
builder.add_edge("find_attributes", "predict_product")
builder.add_edge("predict_product", "get_product_locations")
builder.add_edge("get_product_locations", "generate_answer")

# Completion
builder.add_edge("generate_answer", END)

app = builder.compile()

# Helper function to shorten strings for labels
def shorten_label(label: str, max_length: int = 20) -> str:
    """
    Shortens a label to max_length characters and adds ellipsis if it's too long.

    Args:
        label (str): The full topic label.
        max_length (int): Maximum number of characters to keep before adding "..."

    Returns:
        str: Truncated label.
    """
    label = label.strip().capitalize()
    return label if len(label) <= max_length else label[:max_length].rstrip() + "..."

# ========== 8. Streamlit UI ==========
def set_chat_state(chat_id: str, key: str, value):
    st.session_state[f"{chat_id}:{key}"] = value

def get_chat_state(chat_id: str, key: str, default=None):
    return st.session_state.get(f"{chat_id}:{key}", default)

st.title("🏜️ Explore California - AI Travel Assistant")
st.text("Interact with our AI-powered travel assistant to explore California's best locations and tours!")

# Initialize saved chat sessions
if "saved_chats" not in st.session_state:
    st.session_state.saved_chats = {}

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = "chat_0"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = {}

with st.sidebar:
    st.subheader("⚙️ Settings")
    st.markdown("This app uses LangGraph + RAG with a Mistral LLM via OpenRouter.")
    st.markdown("🔗 [GitHub Repo](https://github.com/LinkedInLearning/applied-AI-and-machine-learning-for-data-practitioners-5932259/blob/main/streamlit-advanced-ai-app.py)")

    # Load a previously saved chat
    st.subheader("🗂️ Chats")
    for chat_id, chat_data in st.session_state.get("saved_chats", {}).items():
        if st.button(chat_data["label"], key=f"load_{chat_id}"):
            st.session_state.chat_history = chat_data["history"]
            st.session_state.last_result = chat_data["result"]
            st.session_state.stored_context = chat_data.get("context", "")
            st.session_state.stored_products = chat_data.get("products", "")
            st.session_state.ml_predicted_product = chat_data.get("predicted_product", "")
            st.session_state.ml_attributes = chat_data.get("attributes", [])
            st.session_state.current_chat_id = chat_id
            st.rerun()

    if st.button("➕ Start new chat"):
        # reset follow-up flag
        st.session_state.followup_flag = False

        # Start a new chat session
        new_chat_id = f"chat_{len(st.session_state.saved_chats) + 1}"
        st.session_state.current_chat_id = new_chat_id
        st.session_state.chat_history = []
        st.session_state.last_result = {}
        st.session_state.stored_context = ""
        st.session_state.stored_products = ""
        st.session_state.ml_predicted_product = ""
        st.session_state.ml_attributes = []
        st.rerun()

query = st.chat_input("Start listing things you like for a personalized recommendation or ask me a question!")

# Handle follow-up clicks
followup_query = st.session_state.pop("selected_followup", None)
actual_query = query or followup_query

if actual_query:
    with st.spinner("🧠 Thinking..."):
        chat_history = st.session_state.get("chat_history", [])
        chat_id = st.session_state.current_chat_id

        # Use followup_mode if we already have stored context
        is_followup = get_chat_state(chat_id, "followup_flag", False)

        result = app.invoke({
            "query": actual_query,
            "chat_history": chat_history,
            "context": get_chat_state(chat_id, "context", ""),
            "products": get_chat_state(chat_id, "products", ""),
            "answer": "",
            "followup_flag": is_followup
        })

        # Store context and results
        set_chat_state(chat_id, "context", result["context"])
        set_chat_state(chat_id, "products", result["products"])
        set_chat_state(chat_id, "predicted_product", result.get("predicted_product", ""))
        set_chat_state(chat_id, "attributes", result.get("attributes", []))
        set_chat_state(chat_id, "last_result", result)
        st.session_state.last_result = result

        # Append to chat history
        st.session_state.chat_history = chat_history + [
            {"role": "user", "content": actual_query},
            {"role": "assistant", "content": result["answer"]}
        ]

        # ✅ Save the chat immediately
        raw_label = result.get("topic_label", "New Chat")
        label = shorten_label(raw_label)

        if "saved_chats" not in st.session_state:
            st.session_state.saved_chats = {}

        st.session_state.saved_chats[chat_id] = {
            "label": label,
            "history": st.session_state.chat_history,
            "result": result,
            "context": result.get("context", ""),
            "products": result.get("products", ""),
            "predicted_product": result.get("predicted_product", ""),
            "attributes": result.get("attributes", []),
        }

        st.rerun()

# Display chat history
if "chat_history" in st.session_state:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# Show context and follow-ups
if "last_result" in st.session_state:
    
    chat_id = st.session_state.current_chat_id
    result = get_chat_state(chat_id, "last_result", {})

    # Only show if there is non-empty result variable
    if result:
        with st.expander("📋 View LLM Result"):
            st.code(json.dumps(result, indent=2), language='json')
        with st.expander("🛠️ View Context"):
            if get_chat_state(chat_id, "predicted_product"):
                st.markdown("**📊 Using ML predicted product info!**")
                st.markdown("**Matched Attributes:**")
                st.markdown(get_chat_state(chat_id, "attributes", []))
            else:
                st.markdown("**📊 Using previously retrieved context for follow-up question!**")

            st.markdown("**Matched Products:**")
            st.markdown(result.get("products", ""))
            st.markdown("**Context:**")
            st.markdown(result.get("context", ""))

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
