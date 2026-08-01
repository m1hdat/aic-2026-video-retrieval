from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config and apply simple environment variable overrides."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    model = config.setdefault("model", {})
    milvus = config.setdefault("milvus", {})
    data = config.setdefault("data", {})

    model["name"] = os.getenv("CLIP_MODEL_NAME", model.get("name"))
    milvus["host"] = os.getenv("MILVUS_HOST", milvus.get("host", "localhost"))
    milvus["port"] = int(os.getenv("MILVUS_PORT", milvus.get("port", 19530)))
    milvus["collection_name"] = os.getenv(
        "COLLECTION_NAME",
        milvus.get("collection_name", "text_image_retrieval"),
    )
    data["image_dir"] = os.getenv("IMAGE_DIR", data.get("image_dir"))
    data["captions_path"] = os.getenv("CAPTIONS_PATH", data.get("captions_path"))
    data["metadata_path"] = os.getenv("METADATA_PATH", data.get("metadata_path"))
    data["embeddings_path"] = os.getenv("EMBEDDINGS_PATH", data.get("embeddings_path"))

    return config


def resolve_project_path(path_value: str | Path) -> Path:
    """Resolve project-relative paths against the repository root."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path
