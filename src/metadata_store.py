from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class MetadataStore:
    def __init__(self, manifest_path: str | Path | None) -> None:
        self.path = Path(manifest_path) if manifest_path else None
        self._by_id: dict[int, dict[str, Any]] = {}
        if self.path and self.path.exists():
            frame = pd.read_parquet(self.path)
            if "global_id" in frame.columns:
                self._by_id = {
                    int(row["global_id"]): row.to_dict()
                    for _, row in frame.iterrows()
                }

    def get(self, global_id: int) -> dict[str, Any]:
        return dict(self._by_id.get(int(global_id), {}))

    def enrich(self, item: dict[str, Any]) -> dict[str, Any]:
        global_id = item.get("id", item.get("global_id"))
        if global_id is None:
            return item
        return {**self.get(int(global_id)), **item}
