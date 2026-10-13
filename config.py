"""All tunable settings in one place. Change numbers here, not inside the logic."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
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
