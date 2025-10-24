# ---------------------------------------------------------
# Secure Streamlit App with Local + Cloud LLMs and Embeddings
# ---------------------------------------------------------

import os
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from ctransformers import AutoModelForCausalLM
from openai import OpenAI

# ---------------------------------------------------------
# 1. Environment & Configuration
# ---------------------------------------------------------

# Prevent Hugging Face tokenizer deadlocks when using forking environments like Streamlit
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables from a .env file in the current working directory
load_dotenv()

# ----------------------------
# Parse optional GPU layer override
# ----------------------------
try:
    GPU_LAYERS = int(os.getenv("GPU_LAYERS", 16))  # Default to 16 if not provided
except ValueError:
    GPU_LAYERS = 16
    st.warning("⚠️ Invalid GPU_LAYERS value in .env — defaulting to 16")

# ----------------------------
# Set app-level secrets
# ----------------------------
STREAMLIT_PASSWORD = os.getenv("STREAMLIT_PASSWORD", "linkedin-learning")
OPEN_ROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")

# ---------------------------------------------------------
# 2. Secure Login Flow
# ---------------------------------------------------------

# Initialize login state if not already present
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "just_authenticated" not in st.session_state:
    st.session_state.just_authenticated = False

# Prompt for password if user is not authenticated
if not st.session_state.authenticated:
    st.title("🔐 Secure Login")
    password = st.text_input("Enter password", type="password")

    # Validate password
    if password == STREAMLIT_PASSWORD:
        st.session_state.authenticated = True
        st.session_state.just_authenticated = True
        st.rerun()
    elif password:
        st.error("Incorrect password")

    # Stop execution until authentication is complete
    st.stop()

# Clear the temporary "just_authenticated" flag after rerun
if st.session_state.just_authenticated:
    st.session_state.just_authenticated = False
    st.rerun()
    
def read_python_file(path: str) -> str:
    """Read the contents of a Python file as a string."""
    with open(path, "r") as f:
        return f.read()

# ---------------------------------------------------------
# 3. Model Loaders (Cached Resources)
# ---------------------------------------------------------

@st.cache_resource(show_spinner="🔄 Loading embedding model...")
def load_local_embedding_model():
    """
    Load a local SentenceTransformer embedding model from path specified in EMBEDDING_MODEL_PATH.
    Caches the model for the session to avoid redundant loading.
    """
    return SentenceTransformer(os.getenv("EMBEDDING_MODEL_PATH"))

@st.cache_resource(show_spinner="🔄 Loading TinyLlama model...")
def load_local_llm():
    """
    Load a local quantized LLM (e.g., TinyLlama) using the `ctransformers` library.
    Respects the GPU_LAYERS value from environment.
    """
    return AutoModelForCausalLM.from_pretrained(
        os.getenv("TINY_LLAMA_MODEL_PATH"),
        model_type="llama",
        gpu_layers=GPU_LAYERS
    )

@st.cache_resource
def get_openrouter_client():
    """
    Instantiate the OpenAI-compatible client for OpenRouter API access.
    Used to interact with hosted models like Mistral.
    """
    return OpenAI(
        api_key=OPEN_ROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )

@st.cache_resource
def get_cloud_llm_response(prompt: str) -> str:
    """
    Send a prompt to OpenRouter's hosted Mistral model and return the generated response.

    Parameters:
    ----------
    prompt : str
        A single prompt string to be sent to the model.

    Returns:
    -------
    str
        Text response from the model (first choice).
    """
    response = open_router_client.chat.completions.create(
        model="mistralai/mistral-small-3.1-24b-instruct:free",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.7
    )
    return response.choices[0].message.content

# ---------------------------------------------------------
# 4. Load Models Once at Runtime
# ---------------------------------------------------------

# Trigger loading of cached models (once per session)
local_llm = load_local_llm()
local_embedding_model = load_local_embedding_model()
open_router_client = get_openrouter_client()

# ---------------------------------------------------------
# 5. Main App UI
# ---------------------------------------------------------

st.title("🧠 Python for AI - Hello World!")

# Text input from user
prompt = st.text_input(
    "📝 Enter a prompt for our local and cloud LLMs",
    placeholder="Start typing your question here..."
)

# Visual separator
st.markdown("---")

# ---------------------------------------------------------
# 6. Model Execution & Output
# ---------------------------------------------------------

if prompt:
    # Show spinner while generating responses
    with st.spinner("🔄 Generating embeddings and responses..."):
        embedding = local_embedding_model.encode(prompt)
        local_llm_response = local_llm(prompt)
        cloud_llm_response = get_cloud_llm_response(prompt)

    # --- Embedding Output (Full Width) ---
    st.markdown("### 📐 Embedding Vector (First 5 Dimensions)")
    st.code(embedding[:5], language="python")

    # --- LLM Comparison Output (Side-by-Side) ---
    st.markdown("---")
    st.markdown("### 🤖 LLM Responses")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 💬 Local LLM - TinyLlama")
        st.write(local_llm_response)

    with col2:
        st.markdown("#### ☁️ Cloud LLM - Mistral")
        st.write(cloud_llm_response)

# ---------------------------------------------------------
# 7. Optional: View Source Code
# ---------------------------------------------------------

with st.expander("🧾 View Source Code (click to expand)"):
    current_file = os.path.abspath(__file__)
    code = read_python_file(current_file)
    st.code(code, language="python")