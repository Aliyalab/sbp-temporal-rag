"""Temporal state checking and timeline overlap validation."""
from __future__ import annotations

from datetime import date
from enum import Enum

from src.models import Passage


class Coverage(Enum):
    COVERED = 1
    BEFORE_COVERAGE = 2


def in_force(passage: Passage, on: date) -> bool:
    """True if the passage rules were active and legally enforceable on the chosen date."""
    if passage.superseded_on is not None:
        return passage.effective_from <= on < passage.superseded_on
    return passage.effective_from <= on


def check_coverage(on: date, coverage_start: date) -> Coverage:
    """Blocks requests seeking knowledge before our oldest loaded document's lifecycle."""
    if on < coverage_start:
        return Coverage.BEFORE_COVERAGE
    return Coverage.COVERED


def find_overlaps(passages: list[Passage]) -> list[str]:
    """Ensures versions of the same provision never overlap in time."""
    overlaps = []
    lineages: dict[str, list[Passage]] = {}
    
    for p in passages:
        lineages.setdefault(p.lineage_id, []).append(p)
        
    for lineage_id, group in lineages.items():
        sorted_group = sorted(group, key=lambda x: x.effective_from)
        for i in range(len(sorted_group) - 1):
            current = sorted_group[i]
            nxt = sorted_group[i+1]
            
            if current.superseded_on is None or current.superseded_on > nxt.effective_from:
                overlaps.append(f"{current.id} overlaps with {nxt.id}")
                
    return overlaps
