"""The question-answering engine.

Design rule: the engine never writes sentences of its own about the rules. It returns
exact quotes from the source passages, so it cannot invent a requirement. When the
evidence is missing, outdated for the date, or the question contains personal data,
it refuses and says why.
"""
from __future__ import annotations

import re
from datetime import date

from src.config import Settings
from src.models import Passage
from src.retrieve import BM25Index, Hit, content_terms, tokenize
from src.schemas import AskResponse, Citation
from src.security import InvalidInput, contains_personal_data, sanitize_question
from src.temporal import Coverage, check_coverage

VERIFY_NOTICE = "Verify against the official SBP text before acting on this."

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+(?=[A-Z(0-9])")


def pick_quotes(question: str, text: str, max_chars: int) -> list[tuple[str, bool]]:
    """Choose the sentences most related to the question. Each returned string is an
    exact substring of `text`. The bool says whether it was cut short."""
    text = text.strip()
    if len(text) <= max_chars:
        return [(text, False)]

    terms = set(content_terms(question))
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    ranked = sorted(
        range(len(sentences)),
        key=lambda i: len(terms & set(tokenize(sentences[i]))),
        reverse=True,
    )
    chosen: list[int] = []
    used = 0
    for i in ranked:
        if not terms & set(tokenize(sentences[i])):
            break
        if used + len(sentences[i]) > max_chars and chosen:
            continue
        chosen.append(i)
        used += len(sentences[i])
        if len(chosen) == 2:
            break
    if not chosen:
        chosen = [0]

    quotes: list[tuple[str, bool]] = []
    for i in sorted(chosen):
        s = sentences[i]
        if len(s) > max_chars:
            cut = s[:max_chars].rsplit(" ", 1)[0]
            quotes.append((cut, True))
        else:
            quotes.append((s, False))
    return quotes


class RagEngine:
    def __init__(self, passages: list[Passage], settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.passages = list(passages)
        self.index = BM25Index(self.passages, self.settings)

    def _refuse(self, reason: str, message: str, on: date) -> AskResponse:
        return AskResponse(status="refused", reason=reason, message=message, transaction_date=on)

    def ask(self, question: str, on: date, today: date | None = None) -> AskResponse:
        s = self.settings
        today = today or date.today()

        try:
            q = sanitize_question(question, s.max_question_chars)
        except InvalidInput as e:
            return self._refuse("invalid_input", str(e), on)

        if contains_personal_data(q):
            # The message deliberately does not repeat what was typed.
            return self._refuse(
                "personal_data",
                "Your question looks like it contains personal or account details "
                "(an ID, account, card or phone number, or an email). Please remove "
                "them and ask the rule question in general terms.",
                on,
            )

        if on > today:
            return self._refuse("future_date", "The transaction date is in the future.", on)

        if check_coverage(on, s.coverage_start) is Coverage.BEFORE_COVERAGE:
            return self._refuse(
                "before_coverage",
                f"The loaded documents only cover rules in force from {s.coverage_start.isoformat()}. "
                f"I cannot reliably tell you what applied on {on.isoformat()}.",
                on,
            )

        hits = self.index.search(q, on)
        citations: list[Citation] = []
        notices: list[str] = []
        for hit in hits:
            citations.extend(self._citations_for(hit, q, notices))

        if not citations:
            return self._refuse(
                "not_found",
                "I could not find this in the documents I have for that date, so I "
                "will not guess. Try rephrasing, or check the official SBP sources.",
                on,
            )

        notices.append(VERIFY_NOTICE)
        return AskResponse(
            status="answered",
            message=(
                f"Found {len(citations)} relevant passage(s) in force on {on.isoformat()}. "
                "The text below is quoted directly from the source and has not been paraphrased."
            ),
            transaction_date=on,
            citations=citations,
            notices=notices,
        )

    def _citations_for(self, hit: Hit, question: str, notices: list[str]) -> list[Citation]:
        p = hit.passage
        out: list[Citation] = []
        for quote, truncated in pick_quotes(question, p.text, self.settings.max_quote_chars):
            if quote not in p.text:  # defence in depth: never emit text that is not in the source
                continue
            out.append(
                Citation(
                    passage_id=p.id, chapter=p.chapter, section=p.section, page=p.page,
                    quote=quote, truncated=truncated, source_url=p.source_url,
                    effective_from=p.effective_from, superseded_on=p.superseded_on,
                    amended_by=p.amended_by,
                )
            )
        if out and p.superseded_on is not None:
            by = f" by {p.amended_by}" if p.amended_by else ""
            notices.append(
                f"{p.chapter} section {p.section} was later amended{by}, effective "
                f"{p.superseded_on.isoformat()}. This answer reflects the rule on your "
                "transaction date."
            )
        return out
