"""
State Bank of Pakistan (SBP) Temporal RAG Engine
Complete Self-Contained Final Project Cloud Build (Zero-Import Architecture)
"""
import streamlit as st
import re
import math
import unicodedata
import json
import os
from datetime import date
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Optional, Literal, Callable
from enum import Enum
from pydantic import BaseModel, Field, ValidationError

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
    rate_limit_requests: int = 30
    rate_limit_window_seconds: int = 60

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

RefusalReason = Literal["invalid_input", "personal_data", "future_date", "before_coverage", "not_found"]

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
    reason: Optional[RefusalReason] = None
    message: str
    transaction_date: date
    citations: list[Citation] = Field(default_factory=list)
    notices: list[str] = Field(default_factory=list)

# ==========================================
# 3. ARCHITECTURE LAYER: SECURITY & SANITIZATION
# ==========================================
class InvalidInput(ValueError):
    """The question cannot be processed."""

def sanitize_question(question: object, max_chars: int) -> str:
    if not isinstance(question, str):
        raise InvalidInput("The question must be text.")
    if len(question) > max_chars * 4:
        raise InvalidInput("The question is too long.")
    text = unicodedata.normalize("NFKC", question)
    text = "".join(ch for ch in text if ch.isspace() or not unicodedata.category(ch).startswith("C"))
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        raise InvalidInput("Please enter a question.")
    if len(text) > max_chars:
        raise InvalidInput("The question is too long.")
    return text

_PERSONAL_PATTERNS = [
    re.compile(r"\b\d{5}-?\d{7}-?\d\b"),                       # CNIC
    re.compile(r"\b(?:\d[ -]?){12,}\d\b"),                     # Cards
    re.compile(r"\b\d{9,}\b"),                                 # Accounts
    re.compile(r"\b[A-Za-z]{2}\d{2}[A-Za-z0-9]{11,30}\b"),     # IBAN
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),                   # Email
]

def contains_personal_data(text: str) -> bool:
    return any(p.search(text) for p in _PERSONAL_PATTERNS)

# ==========================================
# 4. ARCHITECTURE LAYER: TEMPORAL CALCULATIONS
# ==========================================
class Coverage(Enum):
    COVERED = 1
    BEFORE_COVERAGE = 2

def in_force(passage: Passage, on: date) -> bool:
    if passage.superseded_on is not None:
        return passage.effective_from <= on < passage.superseded_on
    return passage.effective_from <= on

def check_coverage(on: date, coverage_start: date) -> Coverage:
    if on < coverage_start:
        return Coverage.BEFORE_COVERAGE
    return Coverage.COVERED

# ==========================================
# 5. ARCHITECTURE LAYER: RETRIEVAL INDEX
# ==========================================
_STOPWORDS = frozenset("a an and are as at be been but by can could did do does for from had has have how i if in into is it its may me my no not of on or our shall should so such than that the their them then there these they this to under until was we were what when where which who whom why will with would you your many much any".split())

def _stem(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token

def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in re.findall(r"[a-z0-9]+", text.lower())]

def content_terms(text: str) -> list[str]:
    seen: dict[str, None] = {}
    for t in tokenize(text):
        if t not in _STOPWORDS and len(t) > 1:
            seen.setdefault(t, None)
    return list(seen)

@dataclass(frozen=True)
class Hit:
    passage: Passage
    score: float
    coverage: float
    matched_terms: tuple[str, ...]

class BM25Index:
    def __init__(self, passages: list[Passage], settings: Settings, k1: float = 1.5, b: float = 0.75) -> None:
        self.settings = settings
        self.passages = list(passages)
        self._k1, self._b = k1, b
        self._tf = [Counter(tokenize(p.text)) for p in self.passages]
        self._len = [sum(c.values()) for c in self._tf]
        n = len(self.passages)
        self._avg_len = (sum(self._len) / n) if n else 0.0
        df: Counter[str] = Counter()
        for c in self._tf:
            df.update(c.keys())
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def _score(self, terms: list[str], i: int) -> float:
        tf, dl = self._tf[i], self._len[i]
        score = 0.0
        for t in terms:
            f = tf.get(t, 0)
            if f == 0: continue
            denom = f + self._k1 * (1 - self._b + self._b * dl / self._avg_len)
            score += self._idf[t] * f * (self._k1 + 1) / denom
        return score

    def search(self, question: str, on: date) -> list[Hit]:
        s = self.settings
        terms = content_terms(question)
        if not terms: return []
        hits: list[Hit] = []
        for i, p in enumerate(self.passages):
            if not in_force(p, on): continue
            matched = tuple(t for t in terms if t in self._tf[i])
            coverage = len(matched) / len(terms)
            if coverage < s.min_term_coverage: continue
            score = self._score(terms, i)
            if score < s.min_bm25_score: continue
            hits.append(Hit(p, score, coverage, matched))
        hits.sort(key=lambda h: (h.score, h.coverage), reverse=True)
        if hits:
            cutoff = hits.score * s.relative_score_cutoff
            hits = [h for h in hits if h.score >= cutoff]
        return hits[:s.top_k]

# ==========================================
# 6. ARCHITECTURE LAYER: EVALUATION ENGINE
# ==========================================
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+(?=[A-Z(0-9])")

def pick_quotes(question: str, text: str, max_chars: int) -> list[tuple[str, bool]]:
    text = text.strip()
    if len(text) <= max_chars: return [(text, False)]
    terms = set(content_terms(question))
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    ranked = sorted(range(len(sentences)), key=lambda i: len(terms & set(tokenize(sentences[i]))), reverse=True)
    chosen: list[int] = []
    used = 0
    for i in ranked:
        if not terms & set(tokenize(sentences[i])): break
        if used + len(sentences[i]) > max_chars and chosen: continue
        chosen.append(i)
        used += len(sentences[i])
        if len(chosen) == 2: break
    if not chosen: 
        chosen = [0] if sentences else []
    quotes: list[tuple[str, bool]] = []
    for i in sorted(chosen):
        s = sentences[i]
        if len(s) > max_chars:
            quotes.append((s[:max_chars], True))
        else:
            quotes.append((s, False))
    return quotes

class RagEngine:
    def __init__(self, passages: list[Passage], settings: Settings) -> None:
        self.settings = settings
        self.passages = list(passages)
        self.index = BM25Index(self.passages, self.settings)

    def _refuse(self, reason: RefusalReason, message: str, on: date) -> AskResponse:
        return AskResponse(status="refused", reason=reason, message=message, transaction_date=on)

    def ask(self, question: str, on: date) -> AskResponse:
        s = self.settings
        try:
            q = sanitize_question(question, s.max_question_chars)
        except InvalidInput as e:
            return self._refuse("invalid_input", str(e), on)

        if contains_personal_data(q):
            return self._refuse("personal_data", "Your question contains sensitive personal identifiers (CNIC/Email/Account Details). Please rephrase in general terms.", on)

        if on > date.today():
            return self._refuse("future_date", "The transaction date is currently in the future.", on)

        if check_coverage(on, s.coverage_start) is Coverage.BEFORE_COVERAGE:
            return self._refuse("before_coverage", f"The loaded database text index only covers items starting from {s.coverage_start.isoformat()}.", on)

        hits = self.index.search(q, on)
        citations: list[Citation] = []
        notices: list[str] = []
        for hit in hits:
            p = hit.passage
            for quote, truncated in pick_quotes(q, p.text, s.max_quote_chars):
                citations.append(Citation(passage_id=p.id, chapter=p.chapter, section=p.section, page=p.page, quote=quote, truncated=truncated, source_url=p.source_url, effective_from=p.effective_from, superseded_on=p.superseded_on, amended_by=p.amended_by))
              if not chosen: 
        chosen = [0] if sentences else []
        
    quotes: list[tuple[str, bool]] = []
    for i in sorted(chosen):
        s = sentences[i]
        if len(s) > max_chars:
            quotes.append((s[:max_chars], True))
        else:
            quotes.append((s, False))
    return quotes

class RagEngine:
    def __init__(self, passages: list[Passage], settings: Settings) -> None:
        self.settings = settings
        self.passages = list(passages)
        self.index = BM25Index(self.passages, self.settings)

    def _refuse(self, reason: RefusalReason, message: str, on: date) -> AskResponse:
        return AskResponse(status="refused", reason=reason, message=message, transaction_date=on)

    def ask(self, question: str, on: date) -> AskResponse:
        s = self.settings
        try:
            q = sanitize_question(question, s.max_question_chars)
        except InvalidInput as e:
            return self._refuse("invalid_input", str(e), on)

        if contains_personal_data(q):
            return self._refuse("personal_data", "Your question contains sensitive personal identifiers (CNIC/Email/Account Details). Please rephrase in general terms.", on)

        if on > date.today():
            return self._refuse("future_date", "The transaction date is currently in the future.", on)

        if check_coverage(on, s.coverage_start) is Coverage.BEFORE_COVERAGE:
            return self._refuse("before_coverage", f"The loaded database text index only covers items starting from {s.coverage_start.isoformat()}.", on)

        hits = self.index.search(q, on)
        citations: list[Citation] = []
        notices: list[str] = []
        for hit in hits:
            p = hit.passage
            for quote, truncated in pick_quotes(q, p.text, s.max_quote_chars):
                citations.append(Citation(passage_id=p.id, chapter=p.chapter, section=p.section, page=p.page, quote=quote, truncated=truncated, source_url=p.source_url, effective_from=p.effective_from, superseded_on=p.superseded_on, amended_by=p.amended_by))
            if p.superseded_on is not None:
                by = f" by {p.amended_by}" if p.amended_by else ""
                notices.append(f"⚠️ Notice: {p.chapter} section {p.section} was subsequently amended{by} effective {p.superseded_on.isoformat()}. Displaying active state for chosen date.")

        if not citations:
            return self._refuse("not_found", "No relevant matching regulatory evidence found active for the selected date. The system refuses to summarize to ensure zero hallucination.", on)

        notices.append("ℹ️ Verification Notice: Validate data against original SBP circulars before enactment execution.")
        return AskResponse(status="answered", message=f"Found {len(citations)} exact source document evidence passage(s) in force on {on.isoformat()}.", transaction_date=on, citations=citations, notices=notices)

# ==========================================
# 7. INTERACTIVE CLIENT WORKSPACE LAYER
# ==========================================
st.set_page_config(
    page_title="State Bank (SBP) Temporal RAG",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown('## 🏦 SBP Temporal RAG Engine')
st.markdown('#### Regulatory Policy Validation Suite with Time-Aware Document Versioning')

DATA_FILE = "data/processed/passages.json"
settings = Settings()

@st.cache_resource
def get_cloud_engine():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)
    passages = [Passage.model_validate(item) for item in raw]
    return RagEngine(passages, settings)

try:
    engine = get_cloud_engine()
    st.success(f"🟢 Connected to Standalone Cloud Engine Index ({len(engine.passages)} provisions loaded seamlessly).")
except Exception as e:
    st.error(f"🔴 Fatal Database Connectivity Error: {e}")
    st.stop()

st.sidebar.markdown("### 🛠️ Evaluation Settings")
selected_date = st.sidebar.date_input("Target Transaction Date", value=date(2026, 6, 1))

question = st.text_input("Ask a regulatory question:", placeholder="e.g., How many days do exporters have to realise export proceeds?")

if st.button("Evaluate Query Timeline", type="primary"):
    if question.strip():
        with st.spinner("Traversing chronological database blocks..."):
            res = engine.ask(question, selected_date)
            
            if res.status == "refused":
                st.error(f"⛔ Engine Refusal Check Triggered: {res.message}")
                st.info(f"**Policy Refusal Profile Code:** `{res.reason}`")
            else:
                st.markdown("### 📑 Unaltered Evidence Citations")
                st.caption(res.message)
                for c in res.citations:
                    st.info(f"“ {c.quote} ”")
                    st.caption(f"📂 Chapter: {c.chapter} - Section: {c.section} - Page: {c.page or 'N/A'} (Enforceable: {c.effective_from} to {c.superseded_on or 'Present'})")
                    st.markdown('---')
                for n in res.notices:
                    st.warning(n)
