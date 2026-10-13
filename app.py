"""Self-contained Serverless SBP Temporal RAG Engine for Cloud Deployment."""
import streamlit as st
from datetime import date

# Import your working core modules natively
from src.config import Settings
from src.store import load_passages
from src.engine import RagEngine

st.set_page_config(
    page_title="State Bank (SBP) Temporal RAG",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-title { font-size: 2.2rem !important; font-weight: 700; margin-bottom: 2px; }
    .subtitle { font-size: 1.05rem !important; color: #64748b; margin-bottom: 25px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🏦 SBP Temporal RAG Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Regulatory Policy Validation Suite with Time-Aware Document Versioning</div>', unsafe_allow_html=True)

DATA_FILE = "data/processed/passages.json"
settings = Settings()

@st.cache_resource
def get_cloud_engine():
    passages = load_passages(DATA_FILE, settings)
    return RagEngine(passages, settings)

try:
    engine = get_cloud_engine()
    st.success(f"🟢 Connected to Autonomous Cloud Engine Index ({len(engine.passages)} passages fully loaded).")
except Exception as e:
    st.error(f"🔴 Core Engine Initialization Fault: {e}")
    st.stop()

st.sidebar.markdown("### 🛠️ Evaluation Settings")
selected_date = st.sidebar.date_input("Target Transaction Date", value=date(2026, 6, 1))

question = st.text_input("Ask a regulatory question:", placeholder="e.g., How many days do exporters have to realise export proceeds?")

if st.button("Evaluate Query Timeline", type="primary"):
    if question.strip():
        with st.spinner("Processing timeline index..."):
            res = engine.ask(question, selected_date)
            
            if res.status == "refused":
                st.error(f"⛔ Refused: {res.message}")
            else:
                st.markdown("### 📑 Unaltered Evidence Citations")
                for c in res.citations:
                    st.info(f"“ {c.quote} ”")
                    st.caption(f"Chapter: {c.chapter} - Section: {c.section} - Enforceable: {c.effective_from} to {c.superseded_on or 'Present'}")
                    st.markdown('---')
                for n in res.notices:
                    st.warning(n)

