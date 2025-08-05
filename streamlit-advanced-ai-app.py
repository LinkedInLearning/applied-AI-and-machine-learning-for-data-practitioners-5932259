# ---------------------------------------------------------
# Explore California - AI Travel App (Advanced LangGraph RAG)
# ---------------------------------------------------------

import os
import pandas as pd
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from typing import TypedDict, List
import joblib

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
    followup_flag: bool
    predicted_product: str
    attributes: List[str]

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
        model_name="mistralai/mistral-small-3.2-24b-instruct:free",
        openai_api_key=OPEN_ROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.7
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
def generate_answer(query: str, context: str, products: str, chat_history: List[dict]) -> dict:
    """Use the LLM to generate an answer incorporating context, products, and chat history."""
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
    """Generate three follow-up questions the user might ask next based on the conversation."""
    prompt = f"""
    Based on the user's question and your response, suggest three follow-up prompts that the user might ask next about travel in California:

    User Question: {query}
    Assistant Answer: {answer}

    Only return the three follow-up prompts separated by newlines.
    """
    result = llm.invoke([HumanMessage(content=prompt)])
    questions = [q.strip("- •\n ") for q in result.content.strip().split("\n") if q.strip()]
    return {"followups": questions}

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
def entry_selector(query: str, followup_flag: bool = False) -> dict:
    """
    Route the flow based on an LLM analysis of the user's query and an optional follow-up flag.

    Args:
        query: The user input text.
        followup: If True, force the route to 'generate_answer'.

    Returns:
        A dict like {"start_node": "find_attributes" | "search_locations" | "generate_answer"}
    """
    if followup_flag:
        return {"start_node": "generate_answer"}

    prompt = f"""
You are an intelligent routing assistant for a travel recommendation AI.

Analyze the following user input and decide what kind of request it is.
Return only one of these options as a single word:
- "find_attributes": if the user is listing or describing their interests or preferences (e.g., "I love hiking and wine tasting" or "Pizza, beaches, mountains, hiking")
- "search_locations": if the user is asking a general question or making an information request (e.g., "What are the best places to visit in California?")
Do not include explanations or extra words.

User input:
"{query}"
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    route = response.content.strip().lower()

    # fallback safeguard
    if route not in {"find_attributes", "search_locations"}:
        route = "generate_answer"

    return {"start_node": route}

# ========== 7. Build LangGraph Workflow ==========
builder = StateGraph(AppState)

# Nodes
builder.add_node("entry_selector", entry_selector)
builder.add_node("find_attributes", find_similar_attributes)
builder.add_node("predict_product", predict_product_from_attributes)
builder.add_node("get_product_locations", get_locations_for_product)
builder.add_node("search_locations", search_locations)
builder.add_node("match_products", match_products)
builder.add_node("generate_answer", generate_answer)
builder.add_node("suggest_followups", suggest_followups)

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
builder.add_edge("generate_answer", "suggest_followups")
builder.add_edge("suggest_followups", END)

app = builder.compile()

# ========== 8. Streamlit UI ==========
st.title("🏜️ Explore California - AI Travel Assistant")
st.text("Interact with our AI-powered travel assistant to explore California's best locations and tours!")
        
with st.sidebar:
    st.subheader("⚙️ Settings")
    st.markdown("This app uses LangGraph + RAG with a Mistral LLM via OpenRouter.")
    st.markdown("🔗 [GitHub Repo](https://github.com/LinkedInLearning/applied-AI-and-machine-learning-for-data-practitioners-5932259)")
    if st.button("🔄 Start Over"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
            # Don't reset authentication state
            st.session_state.authenticated = True
            st.session_state.just_authenticated = True
        st.rerun()

query = st.chat_input("Describe yourself for a personalized recommendation or ask me something about California travel...")

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
            "followup_flag": is_followup
        })

        # Store context and results
        st.session_state.stored_context = result["context"]
        st.session_state.stored_products = result["products"]
        st.session_state.ml_predicted_product = result.get("predicted_product", "")
        st.session_state.ml_attributes = result.get("attributes", [])
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
            if st.session_state.ml_predicted_product:
                st.markdown("**📊 Using ML predicted product info!**")
                st.markdown("**Matched Attributes:**")
                st.markdown(st.session_state.ml_attributes)
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
