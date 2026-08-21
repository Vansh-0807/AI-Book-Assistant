import textwrap
import streamlit as st
from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Book Assistant",
    page_icon="🧠",
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
        #MainMenu {
            visibility: hidden;
        }
        footer {
            visibility: hidden;
        }
        header {
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
            box-shadow:
                0 0 90px rgba(99, 102, 241, 0.45);
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
            box-shadow:
                0 0 120px rgba(168, 85, 247, 0.40);
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
            box-shadow:
                0 0 100px rgba(6, 182, 212, 0.35);
            animation-delay: -7s;
        }
        @keyframes floating {
            0%,
            100% {
                transform:
                    translate3d(0, 0, 0);
            }
            50% {
                transform:
                    translate3d(35px, -45px, 40px);
            }
        }
        @keyframes rotating {
            from {
                rotate: 0deg;
            }
            to {
                rotate: 360deg;
            }
        }
        /* =====================================================
           MAIN CONTENT
        ===================================================== */
        .main-container {
            position: relative;
            z-index: 2;
            max-width: 1400px;
            margin: auto;
            padding-bottom: 40px;
        }
        /* =====================================================
           HERO
        ===================================================== */
        .hero {
            text-align: center;
            padding:
                45px
                20px
                28px
                20px;
        }
        .hero-icon {
            font-size: 58px;
            margin-bottom: 10px;
            filter:
                drop-shadow(
                    0 0 25px
                    rgba(129, 140, 248, 0.55)
                );
            animation:
                iconFloat 4s ease-in-out infinite;
        }
        @keyframes iconFloat {
            0%,
            100% {
                transform:
                    translateY(0px)
                    rotate(0deg);
            }
            50% {
                transform:
                    translateY(-8px)
                    rotate(3deg);
            }
        }
        .hero-title {
            font-size: 58px;
            font-weight: 850;
            letter-spacing: -3px;
            margin: 0;
            background:
                linear-gradient(
                    90deg,
                    #60a5fa,
                    #a78bfa,
                    #f472b6,
                    #60a5fa
                );
            background-size: 300% auto;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation:
                gradientMove 6s linear infinite;
        }
        @keyframes gradientMove {
            to {
                background-position:
                    300% center;
            }
        }
        .hero-subtitle {
            color: #94a3b8;
            font-size: 17px;
            margin-top: 10px;
            margin-bottom: 20px;
        }
        /* =====================================================
           STATUS BADGE
        ===================================================== */
        .status {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding:
                8px
                15px;
            border-radius: 999px;
            background:
                rgba(34, 197, 94, 0.08);
            border:
                1px solid
                rgba(34, 197, 94, 0.22);
            color: #86efac;
            font-size: 13px;
            box-shadow:
                0 0 25px
                rgba(34, 197, 94, 0.05);
        }
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #22c55e;
            box-shadow:
                0 0 12px
                #22c55e;
            animation:
                pulse 1.8s ease-in-out infinite;
        }
        @keyframes pulse {
            0%,
            100% {
                opacity: 1;
            }
            50% {
                opacity: 0.35;
            }
        }
        /* =====================================================
           GLASS CARDS
        ===================================================== */
        .glass-card {
            background:
                rgba(15, 23, 42, 0.58);
            border:
                1px solid
                rgba(148, 163, 184, 0.13);
            border-radius: 22px;
            padding: 22px;
            backdrop-filter:
                blur(18px);
            -webkit-backdrop-filter:
                blur(18px);
            box-shadow:
                0 25px 80px
                rgba(0, 0, 0, 0.32),
                inset 0 1px 0
                rgba(255, 255, 255, 0.04);
            transition:
                transform 0.3s ease,
                border-color 0.3s ease,
                box-shadow 0.3s ease;
        }
        .glass-card:hover {
            transform:
                translateY(-4px);
            border-color:
                rgba(129, 140, 248, 0.32);
            box-shadow:
                0 30px 90px
                rgba(0, 0, 0, 0.42),
                0 0 40px
                rgba(99, 102, 241, 0.07);
        }
        /* =====================================================
           METRICS
        ===================================================== */
        .metric-card {
            text-align: center;
            padding: 20px;
            min-height: 100px;
            border-radius: 20px;
            background:
                rgba(15, 23, 42, 0.62);
            border:
                1px solid
                rgba(148, 163, 184, 0.13);
            backdrop-filter:
                blur(15px);
            box-shadow:
                0 15px 50px
                rgba(0, 0, 0, 0.22);
            transition:
                transform 0.3s ease,
                box-shadow 0.3s ease;
        }
        .metric-card:hover {
            transform:
                translateY(-5px);
            box-shadow:
                0 20px 60px
                rgba(99, 102, 241, 0.14);
        }
        .metric-number {
            font-size: 28px;
            font-weight: 800;
            color: #a78bfa;
            margin-bottom: 5px;
        }
        .metric-label {
            color: #94a3b8;
            font-size: 13px;
        }
        /* =====================================================
           SIDEBAR
        ===================================================== */
        section[data-testid="stSidebar"] {
            background:
                rgba(5, 8, 22, 0.94);
            border-right:
                1px solid
                rgba(148, 163, 184, 0.10);
        }
        .sidebar-logo {
            text-align: center;
            padding:
                20px 0;
        }
        .sidebar-logo-icon {
            font-size: 50px;
            filter:
                drop-shadow(
                    0 0 20px
                    rgba(129, 140, 248, 0.55)
                );
        }
        .sidebar-title {
            font-size: 24px;
            font-weight: 750;
            margin-top: 5px;
            color: #e2e8f0;
        }
        .sidebar-section {
            color: #94a3b8;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            margin-top: 22px;
            margin-bottom: 10px;
        }
        /* =====================================================
           CHAT
        ===================================================== */
        /* Override native chat message containers */
        [data-testid="stChatMessage"] {
            background: rgba(15, 23, 42, 0.72) !important;
            border: 1px solid rgba(148, 163, 184, 0.12) !important;
            padding: 18px 20px !important;
            border-radius: 18px !important;
            margin: 14px 0 !important;
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.18) !important;
        }
        /* Style user messages differently */
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            background: linear-gradient(135deg, rgba(79, 70, 229, 0.30), rgba(59, 130, 246, 0.15)) !important;
            border: 1px solid rgba(99, 102, 241, 0.22) !important;
        }
        /* Keep avatar backgrounds transparent */
        [data-testid="stChatMessageAvatar"] {
            background-color: transparent !important;
        }
        /* =====================================================
           BUTTONS
        ===================================================== */
        .stButton > button {
            border-radius: 12px;
            border:
                1px solid
                rgba(129, 140, 248, 0.30);
            background:
                linear-gradient(
                    135deg,
                    rgba(79, 70, 229, 0.80),
                    rgba(124, 58, 237, 0.80)
                );
            color: white;
            font-weight: 600;
            transition:
                transform 0.25s ease,
                box-shadow 0.25s ease;
        }
        .stButton > button:hover {
            transform:
                translateY(-2px);
            box-shadow:
                0 10px 30px
                rgba(99, 102, 241, 0.25);
        }
        /* =====================================================
           CHAT INPUT
        ===================================================== */
        div[data-testid="stChatInput"] {
            background:
                rgba(15, 23, 42, 0.82);
            border-radius: 18px;
            border:
                1px solid
                rgba(148, 163, 184, 0.14);
            box-shadow:
                0 10px 40px
                rgba(0, 0, 0, 0.25);
        }
        /* =====================================================
           EXPANDER
        ===================================================== */
        div[data-testid="stExpander"] {
            background:
                rgba(15, 23, 42, 0.50);
            border:
                1px solid
                rgba(148, 163, 184, 0.12);
            border-radius: 16px;
        }
        /* =====================================================
           SCROLLBAR
        ===================================================== */
        ::-webkit-scrollbar {
            width: 7px;
        }
        ::-webkit-scrollbar-track {
            background: #050816;
        }
        ::-webkit-scrollbar-thumb {
            background:
                rgba(129, 140, 248, 0.35);
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background:
                rgba(129, 140, 248, 0.55);
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
# LOAD RAG SYSTEM
# ============================================================

@st.cache_resource
def load_rag_system():

    llm = ChatMistralAI(
        model="mistral-small-latest",
        temperature=0.3
    )

    embeddings = MistralAIEmbeddings()

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
                """
You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say:

"I could not find the answer in the document."
"""
            ),

            (
                "human",
                """
Context:

{context}

Question:

{question}
"""
            )
        ]
    )

    return llm, retriever, prompt


llm, retriever, prompt = load_rag_system()


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# MAIN CONTAINER
# ============================================================

st.markdown(
    textwrap.dedent(
        """
        <div class="main-container">
            <div class="hero">
                <div class="hero-icon">
                    🧠
                </div>
                <div class="hero-title">
                    AI Book Assistant
                </div>
                <div class="hero-subtitle">
                    Ask questions. Retrieve knowledge.
                    Get grounded answers.
                </div>
                <div class="status">
                    <span class="status-dot"></span>
                    AI Knowledge Base Ready
                </div>
            </div>
        </div>
        """
    ),
    unsafe_allow_html=True
)





# ============================================================
# WELCOME MESSAGE
# ============================================================

if len(st.session_state.messages) == 0:

    st.markdown(
        textwrap.dedent("""
        <div class="glass-card">
            <h3>
                👋 Welcome to AI Book Assistant
            </h3>
            <p style="color:#94a3b8;">
                Ask questions about your uploaded
                Digital Principles and Computer Organization
                document.
            </p>
            <p style="color:#64748b;">
                The system retrieves relevant document chunks
                using MMR and generates an answer using
                Mistral AI.
            </p>
        </div>
        """),
        unsafe_allow_html=True
    )


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user", avatar="🧑"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant", avatar="🧠"):
            st.markdown(message["content"])

# ============================================================
# CHAT INPUT
# ============================================================

query = st.chat_input(
    "Ask something about Digital Principles & Computer Organization..."
)


# ============================================================
# PROCESS QUERY
# ============================================================

if query:
    
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user", avatar="🧑"):
        st.markdown(query)

    with st.chat_message("assistant", avatar="🧠"):
        
        status = st.status("🔍 Searching knowledge base...", state="running", expanded=False)
        with status:
            docs = retriever.invoke(query)
            st.write(f"Found {len(docs)} relevant chunks.")
            status.update(label="🧠 Generating grounded answer...", state="running")
            
            context = "\n\n".join([doc.page_content for doc in docs])
            
            from langchain_core.output_parsers import StrOutputParser
            chain = prompt | llm | StrOutputParser()
            
            status.update(label="💡 Answering...", state="complete")

        # Stream the response
        stream = chain.stream(
            {
                "context": context,
                "question": query
            }
        )
        answer = st.write_stream(stream)
        
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )
        
        with st.expander("📚 View Retrieved Context"):
            for i, doc in enumerate(docs, start=1):
                st.markdown(f"### Chunk {i}")
                st.write(doc.page_content)
                if i != len(docs):
                    st.divider()
