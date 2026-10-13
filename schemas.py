"""Shapes of what the API returns. Every answer carries its evidence."""
from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field

RefusalReason = Literal[
    "invalid_input",
    "personal_data",
    "future_date",
    "before_coverage",
    "not_found",
]


class Citation(BaseModel):
    passage_id: str
    chapter: str
    section: str
    page: Optional[int] = None
    quote: str = Field(description="Exact text copied from the source passage")
    truncated: bool = Field(default=False, description="True if the quote is cut short")
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
