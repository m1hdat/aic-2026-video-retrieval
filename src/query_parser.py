from __future__ import annotations

import re


TEMPORAL_SEPARATORS = [
    r"\btrước khi\b", r"\bsau đó\b", r"\brồi\b", r"\btiếp theo\b",
    r"\bbefore\b", r"\bafter that\b", r"\bthen\b", r"\bfollowed by\b",
]


def parse_temporal_events(text: str) -> list[str]:
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return []
    pattern = "|".join(TEMPORAL_SEPARATORS)
    parts = [part.strip(" ,.;:-") for part in re.split(pattern, cleaned, flags=re.IGNORECASE)]
    return [part for part in parts if part]


def normalize_events(events: list[str]) -> list[str]:
    return [" ".join(event.split()) for event in events if event and event.strip()]
