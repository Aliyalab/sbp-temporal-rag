"""Load and validate the passage file. Bad data stops the app at startup,
not in the middle of someone's answer."""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse

from pydantic import ValidationError

from src.config import Settings
from src.models import Passage
from src.temporal import find_overlaps


class DataError(RuntimeError):
    """The passage file is missing, malformed or inconsistent."""


def _host_allowed(url: str, allowed: tuple[str, ...]) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https":
        return False
    return any(host == a or host.endswith("." + a) for a in allowed)


def load_passages(path: str | Path, settings: Settings | None = None) -> list[Passage]:
    settings = settings or Settings()
    p = Path(path)
    if not p.is_file():
        raise DataError(f"Passage file not found: {p}")
    if p.stat().st_size > settings.max_data_file_bytes:
        raise DataError("Passage file is larger than the allowed size.")

    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise DataError(f"Passage file could not be read as JSON: {e}") from e
    if not isinstance(raw, list):
        raise DataError("Passage file must contain a JSON list of passages.")

    try:
        passages = [Passage.model_validate(item) for item in raw]
    except ValidationError as e:
        raise DataError(f"A passage is invalid: {e}") from e

    ids = [x.id for x in passages]
    if len(ids) != len(set(ids)):
        raise DataError("Duplicate passage ids found.")

    for x in passages:
        if not _host_allowed(x.source_url, settings.allowed_source_hosts):
            raise DataError(f"{x.id}: source_url must be https on {settings.allowed_source_hosts}.")

    overlaps = find_overlaps(passages)
    if overlaps:
        raise DataError(f"Versions of the same provision overlap in time: {overlaps}")

    return passages
