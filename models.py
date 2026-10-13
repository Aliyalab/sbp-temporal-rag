"""Data representations for the text provisions."""
from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel


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
