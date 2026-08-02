from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import resolve_project_path


def load_manifest(path: str | Path) -> pd.DataFrame:
    resolved = resolve_project_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Không tìm thấy manifest: {resolved}")
    frame = pd.read_parquet(resolved)
    required = {"global_id", "video_id", "frame_id", "feature_row", "clip_feature_relpath"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Manifest thiếu cột: {missing}")
    return frame


def validate_manifest(frame: pd.DataFrame) -> dict[str, bool]:
    return {
        "not_empty": not frame.empty,
        "global_id_unique": bool(frame["global_id"].is_unique),
        "vector_locator_unique": bool(~frame.duplicated(["clip_feature_relpath", "feature_row"]).any()),
        "embedding_dim_512": bool(frame["embedding_dim"].dropna().eq(512).all()) if "embedding_dim" in frame else False,
    }
