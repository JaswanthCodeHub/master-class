# ╔══════════════════════════════════════════════════════════════════╗
# ║        Zyra — HR Help Desk | Zyro Dynamics RAG Challenge        ║
# ╚══════════════════════════════════════════════════════════════════╝

import os
import streamlit as st

# ── Path Detection ────────────────────────────────────────────────────────────
# Kaggle competition path (primary)
KAGGLE_PATH = "/kaggle/input/competitions/niat-masterclass-rag-challenge/zyro-dynamics-hr-corpus/"
# Fallback for Streamlit Cloud (put PDFs in ./pdfs/ in your repo)
LOCAL_PATH  = "./pdfs/"

if os.path.exists(KAGGLE_PATH):
    CORPUS_PATH = KAGGLE_PATH
elif os.path.exists(LOCAL_PATH) and any(
    f.endswith(".pdf") for f in os.listdir(LOCAL_PATH)
):
    CORPUS_PATH = LOCAL_PATH
else:
    CORPUS_PATH = LOCAL_PATH          # will trigger a warning in UI

# ── API Keys / Secrets ────────────────────────────────────────────────────────
def _get_secret(key: str) -> str:
    """Try env var first, then st.secrets, return '' if missing."""
    val = os.environ.get(key, "")
    if not val:
        try:
            val = st.secrets.get(key, "")
        except Exception:
            val = ""
    return val or ""

GROQ_API_KEY      = _get_secret("GROQ_API_KEY")
LANGCHAIN_API_KEY = _get_secret("LANGCHAIN_API_KEY")

if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_API_KEY"]    = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"]    = "zyro-rag-challenge"
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

# ── Model Config ───────────────────────────────────────────────────────────────
LLM_MODEL   = "llama-3.1-8b-instant"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── HR Topic Guardrails ────────────────────────────────────────────────────────
HR_KEYWORDS = [
    "leave", "salary", "compensation", "benefits", "work from home", "wfh",
    "remote", "code of conduct", "conduct", "performance", "review", "appraisal",
    "onboarding", "offboarding", "separation", "travel", "expense", "reimbursement",
    "it security", "data security", "password", "posh", "harassment",
    "sexual harassment", "employee", "handbook", "policy", "hr", "zyro",
    "dynamics", "zyra", "probation", "notice period", "resignation",
    "termination", "overtime", "attendance", "holiday", "sick", "casual leave",
    "maternity", "paternity", "grievance", "complaint", "training",
    "development", "payroll", "increment", "bonus", "allowance", "insurance",
    "medical", "health", "dental", "vision", "provident fund", "pf", "gratuity",
    "promotion", "transfer", "background", "verification", "nda",
]

OOS_RESPONSE = (
    "I'm **Zyra**, your Zyro Dynamics HR assistant. I can only help with "
    "questions about our HR policies — such as leave, compensation, WFH, "
    "code of conduct, performance reviews, IT security, POSH, travel & expenses, "
    "and onboarding/separation.  \n\nCould you rephrase your question in the "
    "context of Zyro Dynamics HR policies?"
)

# ── LangChain Imports ─────────────────────────────────────────────────────────
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ── RAG Pipeline (cached across sessions) ────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_rag_pipeline():
    """Load PDFs → chunk → embed → FAISS → build RAG chain. Cached."""

    # 1. Load
    loader = PyPDFDirectoryLoader(CORPUS_PATH)
    docs   = loader.load()

    # 2. Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    # 3. Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # 4. Vector store + MMR retriever
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever   = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.7},
    )

    # 5. LLM
    llm = ChatGroq(model=LLM_MODEL, temperature=0.1, max_tokens=1024)

    # 6. Prompt
    system_prompt = """You are Zyra, an intelligent HR Help Desk assistant for Zyro Dynamics Pvt. Ltd.
Answer employee questions accurately using ONLY the provided HR policy documents.
Be concise, professional, and helpful.

Rules:
- Use ONLY information from the context provided below
- If the answer is not in the context, clearly say so — do NOT hallucinate
- Where possible, cite the source document name
- Keep answers structured and easy to read

Context:
{context}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human",  "{question}"),
    ])

    def format_docs(docs):
        return "\n\n".join(
            f"[Source: {doc.metadata.get('source', 'Unknown').split('/')[-1]}]\n"
            f"{doc.page_content}"
            for doc in docs
        )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, len(docs), len(chunks)


def is_hr_question(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in HR_KEYWORDS)


def ask_bot(query: str, chain) -> str:
    if not is_hr_question(query):
        return OOS_RESPONSE
    return chain.invoke(query)


# ╔══════════════════════════════════════════════════════════════════╗
# ║                       Streamlit UI                              ║
# ╚══════════════════════════════════════════════════════════════════╝

st.set_page_config(
    page_title="Zyra – HR Help Desk | Zyro Dynamics",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (dark theme) ───────────────────────────────────────────────────
st.markdown("""
<style>
/* Base */
.stApp { background-color: #0d1117; color: #e6edf3; font-family: 'Inter', sans-serif; }

/* Chat messages */
[data-testid="stChatMessage"] {
    background-color: #161b22 !important;
    border: 1px solid #30363d;
    border-radius: 12px;
    margin-bottom: 10px;
    padding: 10px 16px;
}

/* Chat input */
[data-testid="stChatInputTextArea"] {
    background-color: #161b22 !important;
    color: #e6edf3 !important;
    border: 1px solid #30363d !important;
    border-radius: 10px !important;
}

/* Sidebar */
section[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
section[data-testid="stSidebar"] * { color: #e6edf3 !important; }

/* Headings */
h1, h2, h3 { color: #58a6ff !important; }

/* Spinner */
.stSpinner > div { border-top-color: #58a6ff !important; }

/* Policy badges */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72em;
    font-weight: 600;
    margin: 3px 2px;
    letter-spacing: 0.3px;
}
.b-green  { background: #1a3a28; color: #3fb950; border: 1px solid #3fb95040; }
.b-blue   { background: #192642; color: #58a6ff; border: 1px solid #58a6ff40; }
.b-purple { background: #261d42; color: #bc8cff; border: 1px solid #bc8cff40; }
.b-orange { background: #3a2210; color: #f0883e; border: 1px solid #f0883e40; }

/* Status chips */
.chip {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.7em;
    font-weight: 700;
}
.chip-ok  { background: #1a3a28; color: #3fb950; }
.chip-off { background: #3a1a1a; color: #f85149; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 Zyra")
    st.caption("HR Help Desk · Zyro Dynamics Pvt. Ltd.")
    st.divider()

    # Status
    groq_status  = '<span class="chip chip-ok">✓ Connected</span>' if GROQ_API_KEY      else '<span class="chip chip-off">✗ Missing</span>'
    ls_status    = '<span class="chip chip-ok">✓ Active</span>'    if LANGCHAIN_API_KEY else '<span class="chip chip-off">✗ Disabled</span>'
    path_status  = '<span class="chip chip-ok">✓ Found</span>'     if os.path.exists(CORPUS_PATH) else '<span class="chip chip-off">✗ Not found</span>'

    st.markdown("**🔑 API Status**")
    st.markdown(f"Groq API: {groq_status}", unsafe_allow_html=True)
    st.markdown(f"LangSmith: {ls_status}", unsafe_allow_html=True)
    st.markdown(f"Corpus: {path_status}", unsafe_allow_html=True)

    st.divider()
    st.markdown("**📂 Knowledge Base** *(11 PDFs)*")
    st.markdown("""
<span class='badge b-blue'>Company Profile</span>
<span class='badge b-green'>Employee Handbook</span>
<span class='badge b-green'>Leave Policy</span>
<span class='badge b-blue'>WFH Policy</span>
<span class='badge b-purple'>Code of Conduct</span>
<span class='badge b-orange'>Performance Review</span>
<span class='badge b-green'>Compensation &amp; Benefits</span>
<span class='badge b-purple'>IT &amp; Data Security</span>
<span class='badge b-orange'>POSH Policy</span>
<span class='badge b-blue'>Onboarding &amp; Separation</span>
<span class='badge b-green'>Travel &amp; Expense</span>
""", unsafe_allow_html=True)

    st.divider()
    st.markdown("**⚙️ Pipeline**")
    st.caption(f"🧠 LLM: `{LLM_MODEL}`")
    st.caption(f"📐 Embeddings: `all-MiniLM-L6-v2`")
    st.caption(f"🔍 Retriever: FAISS · MMR · k=5")

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("## 🤖 Zyra — HR Help Desk")
st.caption("Powered by Zyro Dynamics HR Policy Corpus · Built with LangChain + FAISS + Groq")

# Corpus path warning
if not os.path.exists(CORPUS_PATH):
    st.warning(
        f"⚠️ PDF corpus not found at `{CORPUS_PATH}`.  \n"
        "For **Streamlit Cloud**: add your 11 PDFs to a `pdfs/` folder in your repo.  \n"
        "For **Kaggle**: make sure the dataset is attached to your notebook.",
        icon="📁",
    )

# Build pipeline
with st.spinner("⚙️ Loading HR knowledge base — this takes ~30 sec the first time..."):
    try:
        rag_chain, n_docs, n_chunks = build_rag_pipeline()
        st.success(
            f"✅ Knowledge base ready — {n_docs} pages · {n_chunks} chunks indexed.",
            icon="📚",
        )
    except Exception as e:
        st.error(f"❌ Pipeline build failed: {e}")
        st.stop()

# ── Chat history init ─────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 Hi! I'm **Zyra**, your HR assistant at Zyro Dynamics.  \n"
                "I can answer questions about **leave policies, compensation, WFH, "
                "code of conduct, performance reviews, IT security, POSH, "
                "travel & expenses, onboarding, and separation**.  \n\n"
                "What would you like to know?"
            ),
        }
    ]

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Input ─────────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about an HR policy..."):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching policies..."):
            try:
                response = ask_bot(prompt, rag_chain)
            except Exception as e:
                response = f"⚠️ Error generating response: `{e}`"
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
