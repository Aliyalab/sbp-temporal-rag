"""Keyword search (BM25) that only ever sees rules in force on the chosen date."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date

from src.config import Settings
from src.models import Passage
from src.temporal import in_force

_STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have how i if in
    into is it its may me my no not of on or our shall should so such than that the their them
    then there these they this to under until was we were what when where which who whom why will
    with would you your many much any""".split()
)


def _stem(token: str) -> str:
    """Very light plural handling so 'remittances' matches 'remittance'."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in re.findall(r"[a-z0-9]+", text.lower())]


def content_terms(text: str) -> list[str]:
    """Meaningful, de-duplicated words of a question (order kept)."""
    seen: dict[str, None] = {}
    for t in tokenize(text):
        if t not in _STOPWORDS and len(t) > 1:
            seen.setdefault(t, None)
    return list(seen)


@dataclass(frozen=True)
class Hit:
    passage: Passage
    score: float
    coverage: float          # share of the question's meaningful words found in the passage
    matched_terms: tuple[str, ...]


class BM25Index:
    def __init__(self, passages: list[Passage], settings: Settings | None = None,
                 k1: float = 1.5, b: float = 0.75) -> None:
        self.settings = settings or Settings()
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
            if f == 0:
                continue
            denom = f + self._k1 * (1 - self._b + self._b * dl / self._avg_len)
            score += self._idf[t] * f * (self._k1 + 1) / denom
        return score

    def search(self, question: str, on: date, top_k: int | None = None) -> list[Hit]:
        """Return the best passages that were in force on `on` and clearly match.

        Superseded or not-yet-effective passages are skipped BEFORE scoring, so an
        outdated rule can never be returned. Weak matches are dropped, which is what
        lets the system answer 'not found' instead of guessing.
        """
        s = self.settings
        terms = content_terms(question)
        if not terms:
            return []

        hits: list[Hit] = []
        for i, p in enumerate(self.passages):
            if not in_force(p, on):
                continue
            matched = tuple(t for t in terms if t in self._tf[i])
            coverage = len(matched) / len(terms)
            if coverage < s.min_term_coverage:
                continue
            score = self._score(terms, i)
            if score < s.min_bm25_score:
                continue
            hits.append(Hit(p, score, coverage, matched))

        hits.sort(key=lambda h: (h.score, h.coverage), reverse=True)
        if hits:
            cutoff = hits[0].score * s.relative_score_cutoff
            hits = [h for h in hits if h.score >= cutoff]
        return hits[: (top_k or s.top_k)]
