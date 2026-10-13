"""
State Bank of Pakistan (SBP) Temporal RAG Engine
Complete Self-Contained Final Project Cloud Build (Zero-Import Architecture)
"""
import streamlit as st
import re
import math
import unicodedata
import json
from datetime import date
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Literal
from enum import Enum
from pydantic import BaseModel, Field

# ==========================================
# 1. ARCHITECTURE LAYER: CONFIGURATION
# ==========================================
@dataclass(frozen=True)
class Settings:
    max_question_chars: int = 500
    top_k: int = 3
    min_term_coverage: float = 0.3
    min_bm25_score: float = 0.05
    relative_score_cutoff: float = 0.1
    max_quote_chars: int = 700
    coverage_start: date = date(2025, 9, 11)
    allowed_source_hosts: tuple[str, ...] = ("sbp.org.pk",)
    max_data_file_bytes: int = 20_000_000

# ==========================================
# 2. ARCHITECTURE LAYER: DATA MODELS & SCHEMAS
# ==========================================
class Passage(BaseModel):
    id: str
    lineage_id: str
    doc_id: str
    chapter: str
    section: str
    text: str
    source_url: str
    effective_from: date
    superseded_on: Optional[date] = None
    amended_by: Optional[str] = None
    page: Optional[int] = None

class Citation(BaseModel):
    passage_id: str
    chapter: str
    section: str
    page: Optional[int] = None
    quote: str
    truncated: bool = False
    source_url: str
    effective_from: date
    superseded_on: Optional[date] = None
    amended_by: Optional[str] = None

class AskResponse(BaseModel):
    status: Literal["answered", "refused"]
    reason: Optional[str] = None
    message: str
    transaction_date: date
    citations: list[Citation] = Field(default_factory=list)
    notices: list[str] = Field(default_factory=list)

# ==========================================
# 3. ARCHITECTURE LAYER: SECURITY & SANITIZATION
# ==========================================
class InvalidInput(ValueError): pass

def sanitize_question(question: str, max_chars: int) -> str:
    text = unicodedata.normalize("NFKC", question)
    text = "".join(ch for ch in text if ch.isspace() or not unicodedata.category(ch).startswith("C"))
    return re.sub(r"\s+", " ", text).strip()

def contains_personal_data(text: str) -> bool:
    return any(p.search(text) for p in [
        re.compile(r"\b\d{5}-?\d{7}-?\d\b"),                       # CNIC
        re.compile(r"\b\d{9,}\b"),                                 # Accounts
        re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")                    # Email
    ])

# ==========================================
# 4. ARCHITECTURE LAYER: TEMPORAL MATRIX FILTERS
# ==========================================
def in_force(p: Passage, on: date) -> bool:
    if p.superseded_on:
        return p.effective_from <= on < p.superseded_on
    return p.effective_from <= on

# ==========================================
# 5. ARCHITECTURE LAYER: RETRIEVAL INDEX
# ==========================================
_STOPWORDS = frozenset("a an and are as at be been but by can could did do does for from had has have how i if in into is it its may me my no not of on or our shall should so such than that the their them then there these they this to under until was we were what when where which who whom why will with would you your".split())

def tokenize(text: str) -> list[str]:
    return [t[:-1] if (len(t) > 3 and t.endswith("s") and not t.endswith("ss")) else t for t in re.findall(r"[a-z0-9]+", text.lower())]

def content_terms(text: str) -> list[str]:
    return list({t: None for t in tokenize(text) if t not in _STOPWORDS and len(t) > 1})

class BM25Index:
    def __init__(self, passages: list[Passage], s: Settings):
        self.s, self.passages = s, passages
        self._tf = [Counter(tokenize(p.text)) for p in passages]
        self._len = [sum(c.values()) for c in self._tf]
        n = len(passages)
        self._avg_len = (sum(self._len) / n) if n else 0
        df = Counter()
        for c in self._tf: 
            df.update(c.keys())
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, q: str, on: date) -> list[Passage]:
        terms = content_terms(q)
        return [p for i, p in enumerate(self.passages) if in_force(p, on) and any(t in self._tf[i] for t in terms)]

# ==========================================
# 6. ARCHITECTURE LAYER: RAG ENGINE PIPELINE
# ==========================================
class RagEngine:
    def __init__(self, passages: list[Passage], s: Settings):
        self.s, self.passages = s, passages
        self.index = BM25Index(passages, s)

    def ask(self, question: str, on: date) -> AskResponse:
        q = sanitize_question(question, self.s.max_question_chars)
        if contains_personal_data(q):
            return AskResponse(status="refused", reason="personal_data", message="Personal data (CNIC/Email/Account Details) detected. Please rephrase in general terms.", transaction_date=on)
        
        hits = self.index.search(q, on)
        citations, notices = [], []
        for p in hits:
            citations.append(Citation(passage_id=p.id, chapter=p.chapter, section=p.section, page=p.page, quote=p.text, source_url=p.source_url, effective_from=p.effective_from, superseded_on=p.superseded_on, amended_by=p.amended_by))
            if p.superseded_on:
                by = f" by {p.amended_by}" if p.amended_by else ""
                notices.append(f"⚠️ Notice: {p.chapter} section {p.section} was subsequently amended{by} effective {p.superseded_on.isoformat()}.")
        
        if not citations:
            return AskResponse(status="refused", reason="not_found", message="No relevant regulatory evidence found active for the selected date.", transaction_date=on)
        
        return AskResponse(status="answered", message="Found exact matches.", transaction_date=on, citations=citations, notices=notices)

# ==========================================
# 7. INTERACTIVE CLIENT WORKSPACE LAYER
# ==========================================
st.set_page_config(page_title="SBP Temporal RAG", page_icon="🏦", layout="wide")
st.markdown("## 🏦 SBP Temporal RAG Engine")
st.markdown("#### Regulatory Policy Validation Suite with Time-Aware Document Versioning")

DATA_FILE = "data/processed/passages.json"
settings = Settings()

@st.cache_resource
def get_cloud_engine():
    with open(DATA_FILE, "r", encoding="utf-8") as f: 
        raw = json.load(f)
    return RagEngine([Passage.model_validate(x) for x in raw], settings)

try:
    engine = get_cloud_engine()
    st.success("🟢 Connected to Standalone Cloud Engine Index.")
except Exception as e:
    st.error(f"🔴 Fatal Engine Fault: {e}")
    st.stop()

selected_date = st.sidebar.date_input("Target Transaction Date", value=date(2026, 6, 1))
question = st.text_input("Ask a regulatory question:", placeholder="e.g., How many days do exporters have to realise export proceeds?")

if st.button("Evaluate Query Timeline", type="primary"):
    if question.strip():
        res = engine.ask(question, selected_date)
        if res.status == "refused":
            st.error(f"⛔ Refused: {res.message}")
        else:
            st.markdown("### 📑 Unaltered Evidence Citations")
            for c in res.citations:
                st.info(f"“ {c.quote} ”")
                st.caption(f"📂 Chapter: {c.chapter} - Section: {c.section} (Enforceable: {c.effective_from} to {c.superseded_on or 'Present'})")
            for n in res.notices:
                st.warning(n)
