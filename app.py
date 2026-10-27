import warnings
warnings.filterwarnings('ignore')
import textwrap
import os
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ============================================================
# ENVIRONMENT & STREAMLIT CLOUD SECRETS
# ============================================================

load_dotenv()

# Read secrets from Streamlit Cloud dashboard if deployed
try:
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    elif "GEMINI_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass



import io
import re
import urllib.parse
import requests
from PIL import Image
import streamlit.components.v1 as components

# ============================================================
# DIAGRAM & IMAGE GENERATION HELPERS
# ============================================================

def render_mermaid(diagram_code: str, height: int = 360):
    """Renders a Mermaid.js diagram directly inside an interactive HTML container."""
    html_code = f"""
    <!DOCTYPE html>
    <html>
      <head>
        <script type="module">
          import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
          mermaid.initialize({{ 
              startOnLoad: true, 
              theme: 'dark',
              securityLevel: 'loose'
          }});
        </script>
        <style>
          body {{
            background: transparent;
            margin: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            overflow-x: auto;
          }}
          .mermaid svg {{
            max-width: 100%;
            height: auto;
          }}
        </style>
      </head>
      <body>
        <div class="mermaid">
          {diagram_code}
        </div>
      </body>
    </html>
    """
    components.html(html_code, height=height, scrolling=True)

def display_response_with_diagrams(content: str):
    """
    Parses content for ```mermaid ... ``` code blocks.
    Renders text with st.markdown and diagrams with render_mermaid().
    """
    mermaid_regex = r"```mermaid\s*([\s\S]*?)\s*```"
    parts = re.split(mermaid_regex, content)
    for i, part in enumerate(parts):
        if i % 2 == 0:
            if part.strip():
                st.markdown(part)
        else:
            if part.strip():
                render_mermaid(part.strip())

def is_image_request(user_query: str) -> bool:
    """Detects if user explicitly asked for an image or illustration."""
    keywords = [
        "image", "picture", "photo", "illustration", "draw",
        "generate image", "create image", "show picture"
    ]
    q = user_query.lower()
    return any(k in q for k in keywords)

def generate_visual_image(prompt_text: str):
    """
    Generates an AI visual for technical and computer science concepts.
    Tries Google Imagen 3 first, and automatically falls back to Pollinations AI
    so the user is guaranteed an image with zero downtime or quota issues.
    """
    # 1. Try Google Imagen 3 if available
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        result = client.models.generate_images(
            model="imagen-3.0-generate-002",
            prompt=f"Educational technical visual: {prompt_text}, high detail, 3D computer engineering render, clean background",
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio="16:9",
                output_mime_type="image/jpeg",
            )
        )
        if result and result.generated_images:
            img_bytes = result.generated_images[0].image.image_bytes
            return Image.open(io.BytesIO(img_bytes))
    except Exception:
        pass

    # 2. Resilient fallback via high-quality Pollinations AI
    try:
        encoded_prompt = urllib.parse.quote(f"Computer science concept diagram: {prompt_text}, 4k, crisp, educational illustration")
        image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=576&nologo=true&seed=42"
        resp = requests.get(image_url, timeout=20)
        if resp.status_code == 200:
            return Image.open(io.BytesIO(resp.content))
    except Exception:
        pass

    return None

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Book Assistant",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    textwrap.dedent(
        """
        <style>
        /* =====================================================
           GLOBAL
        ===================================================== */
        .stApp {
            background:
                radial-gradient(
                    circle at 10% 15%,
                    rgba(99, 102, 241, 0.16),
                    transparent 25%
                ),
                radial-gradient(
                    circle at 90% 15%,
                    rgba(168, 85, 247, 0.15),
                    transparent 28%
                ),
                radial-gradient(
                    circle at 50% 90%,
                    rgba(6, 182, 212, 0.10),
                    transparent 30%
                ),
                #050816;
            color: #f8fafc;
        }
        /* Remove Streamlit branding */
        #MainMenu, footer, header {
            visibility: hidden;
        }
        
        /* =====================================================
           ANIMATED 3D BACKGROUND
        ===================================================== */
        .background-scene {
            position: fixed;
            width: 100vw;
            height: 100vh;
            top: 0;
            left: 0;
            overflow: hidden;
            pointer-events: none;
            z-index: 0;
        }
        .orb {
            position: absolute;
            border-radius: 50%;
            filter: blur(1px);
            opacity: 0.45;
            animation:
                floating 12s ease-in-out infinite,
                rotating 18s linear infinite;
            transform-style: preserve-3d;
        }
        .orb-one {
            width: 190px;
            height: 190px;
            left: 8%;
            top: 12%;
            background:
                radial-gradient(
                    circle at 30% 30%,
                    rgba(99, 102, 241, 0.9),
                    rgba(99, 102, 241, 0.05) 70%
                );
            box-shadow: 0 0 90px rgba(99, 102, 241, 0.45);
            animation-delay: 0s;
        }
        .orb-two {
            width: 240px;
            height: 240px;
            right: 7%;
            top: 18%;
            background:
                radial-gradient(
                    circle at 30% 30%,
                    rgba(168, 85, 247, 0.9),
                    rgba(168, 85, 247, 0.04) 70%
                );
            box-shadow: 0 0 120px rgba(168, 85, 247, 0.40);
            animation-delay: -4s;
        }
        .orb-three {
            width: 140px;
            height: 140px;
            left: 48%;
            bottom: 10%;
            background:
                radial-gradient(
                    circle at 30% 30%,
                    rgba(6, 182, 212, 0.9),
                    rgba(6, 182, 212, 0.04) 70%
                );
            box-shadow: 0 0 100px rgba(6, 182, 212, 0.35);
            animation-delay: -7s;
        }
        @keyframes floating {
            0%, 100% { transform: translate3d(0, 0, 0); }
            50% { transform: translate3d(35px, -45px, 40px); }
        }
        @keyframes rotating {
            from { rotate: 0deg; }
            to { rotate: 360deg; }
        }

        /* =====================================================
           HERO SECTION
        ===================================================== */
        .hero {
            text-align: center;
            padding: 35px 20px 20px 20px;
        }
        .hero-icon {
            font-size: 52px;
            margin-bottom: 8px;
            filter: drop-shadow(0 0 25px rgba(129, 140, 248, 0.55));
            animation: iconFloat 4s ease-in-out infinite;
        }
        @keyframes iconFloat {
            0%, 100% { transform: translateY(0px) rotate(0deg); }
            50% { transform: translateY(-8px) rotate(3deg); }
        }
        .hero-title {
            font-size: 52px;
            font-weight: 850;
            letter-spacing: -2px;
            margin: 0;
            background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6, #60a5fa);
            background-size: 300% auto;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: gradientMove 6s linear infinite;
        }
        @keyframes gradientMove {
            to { background-position: 300% center; }
        }
        .hero-subtitle {
            color: #94a3b8;
            font-size: 16px;
            margin-top: 8px;
            margin-bottom: 16px;
        }

        /* STATUS BADGE */
        .status {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 14px;
            border-radius: 999px;
            background: rgba(34, 197, 94, 0.08);
            border: 1px solid rgba(34, 197, 94, 0.22);
            color: #86efac;
            font-size: 13px;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #22c55e;
            box-shadow: 0 0 12px #22c55e;
            animation: pulse 1.8s ease-in-out infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.35; }
        }

        /* GLASS CARD */
        .glass-card {
            background: rgba(15, 23, 42, 0.58);
            border: 1px solid rgba(148, 163, 184, 0.13);
            border-radius: 20px;
            padding: 20px;
            backdrop-filter: blur(18px);
            margin-bottom: 20px;
        }

        /* CHAT STYLING */
        [data-testid="stChatMessage"] {
            background: rgba(15, 23, 42, 0.72) !important;
            border: 1px solid rgba(148, 163, 184, 0.12) !important;
            padding: 16px 20px !important;
            border-radius: 18px !important;
            margin: 12px 0 !important;
        }
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            background: linear-gradient(135deg, rgba(79, 70, 229, 0.25), rgba(59, 130, 246, 0.15)) !important;
            border: 1px solid rgba(99, 102, 241, 0.22) !important;
        }

        /* BUTTONS & CHAT INPUT */
        .stButton > button {
            border-radius: 12px;
            border: 1px solid rgba(129, 140, 248, 0.30);
            background: rgba(30, 41, 59, 0.7);
            color: #cbd5e1;
            transition: all 0.2s ease;
            width: 100%;
        }
        .stButton > button:hover {
            border-color: rgba(129, 140, 248, 0.6);
            color: #ffffff;
            background: rgba(79, 70, 229, 0.4);
        }
        div[data-testid="stChatInput"] {
            background: rgba(15, 23, 42, 0.82);
            border-radius: 18px;
            border: 1px solid rgba(148, 163, 184, 0.14);
        }
        </style>

        <div class="background-scene">
            <div class="orb orb-one"></div>
            <div class="orb orb-two"></div>
            <div class="orb orb-three"></div>
        </div>
        """
    ),
    unsafe_allow_html=True
)


# ============================================================
# LOAD RAG SYSTEM (SILENCED DEFAULT LOADING SPINNER)
# ============================================================

@st.cache_resource(show_spinner=False)
def load_rag_system():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        timeout=30
    )

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    vectorstore = Chroma(
        persist_directory="chroma-db",
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "lambda_mult": 0.5
        }
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert AI assistant for Digital Principles & Computer Organization.
Use ONLY the provided context to answer the question.
If the answer is not present in the context, say:
"I could not find the answer in the document."

### Diagram Generation Guidelines:
When explaining concepts involving multi-step processes, computer architectures, hardware layouts, state transitions, or execution flows (such as CPU pipelining, instruction cycle, memory hierarchy, cache operation, K-maps, or logic flow):
1. Provide a clear, thorough explanation.
2. Include a valid, clean Mermaid diagram inside a ```mermaid ... ``` code block.
3. Supported diagram types:
   - Flowcharts / Block Diagrams: `graph TD` (top-down) or `graph LR` (left-to-right)
   - Sequence Diagrams: `sequenceDiagram`
   - State Diagrams: `stateDiagram-v2`
4. Always wrap labels with special characters in double quotes: e.g. id["Instruction Fetch (IF)"]
5. Keep the diagram clean, readable, and directly aligned with the context."""
            ),
            (
                "human",
                """Context:
{context}

Question:
{question}"""
            )
        ]
    )

    return llm, retriever, prompt


# Validate API key before initializing to avoid raw crashes on Streamlit Cloud
if not os.getenv("GOOGLE_API_KEY"):
    st.error(
        "🔑 **Google Gemini API Key is missing!**\n\n"
        "If you are deployed on **Streamlit Cloud**:\n"
        "1. Click **Manage app** in the bottom right corner of this screen.\n"
        "2. Click the three dots `⋮` -> **Settings** -> **Secrets**.\n"
        "3. Paste your key in this format:\n"
        "```toml\n"
        "GOOGLE_API_KEY = \"your_gemini_api_key_here\"\n"
        "```\n"
        "4. Click **Save** and the app will start working immediately!"
    )
    st.stop()

if not os.path.exists("chroma-db"):
    st.error(
        "📚 **Database `chroma-db` not found on server!**\n\n"
        "Make sure your `chroma-db/` folder is committed to your GitHub repository."
    )
    st.stop()

# Quietly initialize RAG system
llm, retriever, prompt = load_rag_system()


# ============================================================
# SESSION STATE
# ============================================================


with st.sidebar:
    st.markdown("### Controls")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.cache_resource.clear()
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

def select_suggestion(prompt_text):
    st.session_state.pending_prompt = prompt_text


# ============================================================
# HERO HEADER
# ============================================================

st.markdown(
    textwrap.dedent(
        """
        <div class="hero">
            <div class="hero-icon">🧠</div>
            <div class="hero-title">AI Book Assistant</div>
            <div class="hero-subtitle">Ask questions. Retrieve knowledge. Get grounded answers.</div>
            <div class="status">
                <span class="status-dot"></span>
                AI Knowledge Base Ready
            </div>
        </div>
        """
    ),
    unsafe_allow_html=True
)


# ============================================================
# SUGGESTED PROMPTS (PERSISTENT & ALWAYS CLICKABLE)
# ============================================================

st.markdown("<p style='color:#94a3b8; font-size:13px;'>Suggested questions & visuals:</p>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    st.button(
        "💡 Explain Karnaugh maps",
        key="btn_kmap",
        on_click=select_suggestion,
        args=("Explain Karnaugh maps",),
        use_container_width=True
    )
with col2:
    st.button(
        "⚙️ Instruction pipelining",
        key="btn_pipeline",
        on_click=select_suggestion,
        args=("Explain instruction pipelining with a flow diagram",),
        use_container_width=True
    )
with col3:
    st.button(
        "🔌 Cache memory hierarchy",
        key="btn_cache",
        on_click=select_suggestion,
        args=("Explain memory hierarchy from registers to disk with a diagram",),
        use_container_width=True
    )
# with col4:
#     st.button(
#         "🎨 Draw CPU architecture",
#         key="btn_img",
#         on_click=select_suggestion,
#         args=("Generate an image of a modern computer CPU architecture and explain it",),
#         use_container_width=True
#     )


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user", avatar="🧑"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant", avatar="🧠"):
            display_response_with_diagrams(message["content"])
            if "image" in message and message["image"]:
                st.image(message["image"], caption="Generated Technical Illustration", use_container_width=True)


# ============================================================
# CHAT INPUT & PROMPT HANDLING
# ============================================================

user_input = st.chat_input("Ask something about Digital Principles & Computer Organization...")
query = user_input or st.session_state.pending_prompt

# Reset the pending prompt so each click fires exactly once
if st.session_state.pending_prompt:
    st.session_state.pending_prompt = None


# ============================================================
# PROCESS QUERY
# ============================================================

if query:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user", avatar="🧑"):
        st.markdown(query)

    # Process Assistant Response
    with st.chat_message("assistant", avatar="🧠"):
        try:
            with st.spinner("Thinking..."):
                # Retrieve relevant documents quietly in backend
                docs = retriever.invoke(query)
                context = "\n\n".join([doc.page_content for doc in docs])
                chain = prompt | llm | StrOutputParser()
                
                answer = None
                last_error = None
                import time

                for attempt in range(1, 4):
                    try:
                        answer = chain.invoke({"context": context, "question": query})
                        break
                    except Exception as e:
                        last_error = e
                        err_text = str(e)
                        if "503" in err_text or "UNAVAILABLE" in err_text or "high demand" in err_text or "timed out" in err_text:
                            time.sleep(2 * attempt)
                        else:
                            raise e

                if answer is None:
                    raise last_error

            # Display the synthesized answer + any diagrams
            display_response_with_diagrams(answer)

            # Check if user explicitly requested an image or visual illustration
            generated_img = None
            if is_image_request(query):
                with st.spinner("🎨 Generating technical illustration..."):
                    generated_img = generate_visual_image(query)
                    if generated_img:
                        st.image(generated_img, caption="AI Generated Technical Illustration", use_container_width=True)

            # Append assistant message to session state (including image if generated)
            msg_data = {"role": "assistant", "content": answer}
            if generated_img:
                msg_data["image"] = generated_img
            st.session_state.messages.append(msg_data)
        except Exception as e:
            st.error(f"⚠️ Error: {e}")