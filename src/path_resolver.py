from __future__ import annotations

from pathlib import Path


class PathResolver:
    def __init__(self, part_roots: dict[str, str] | None = None) -> None:
        self.part_roots = {key: Path(value) for key, value in (part_roots or {}).items()}

    def resolve(self, source_part: str | None, relative_path: str | None, absolute_path: str | None = None) -> str | None:
        if absolute_path:
            candidate = Path(absolute_path)
            if candidate.exists():
                return str(candidate.resolve())

        if not relative_path:
            return None
        rel = Path(relative_path)
        if rel.is_absolute() and rel.exists():
            return str(rel.resolve())

        root = self.part_roots.get(str(source_part))
        if root:
            candidate = root / rel
            if candidate.exists():
                return str(candidate.resolve())
        return None
