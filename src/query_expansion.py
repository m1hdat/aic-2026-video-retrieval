from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExpandedQuery:
    text: str
    weight: float = 1.0


class QueryExpander:
    """Deterministic CLIP prompt expansion; no external API dependency."""

    def __init__(self, templates: list[str] | None = None, max_variants: int = 3) -> None:
        self.templates = templates or ["{query}", "a photo of {query}", "a video frame showing {query}"]
        self.max_variants = max(1, int(max_variants))

    def expand(self, query: str) -> list[ExpandedQuery]:
        cleaned = " ".join((query or "").split())
        if not cleaned:
            return []

        seen: set[str] = set()
        variants: list[ExpandedQuery] = []
        for index, template in enumerate(self.templates[: self.max_variants]):
            text = template.format(query=cleaned).strip()
            if not text or text.lower() in seen:
                continue
            seen.add(text.lower())
            variants.append(ExpandedQuery(text=text, weight=1.0 if index == 0 else 0.9))
        return variants
