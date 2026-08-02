from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# ===========================
# Project paths
# ===========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


# ===========================
# Helpers
# ===========================

def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


# ===========================
# Load config
# ===========================

def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy config: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    # -------------------------
    # Sections
    # -------------------------

    model = config.setdefault("model", {})
    milvus = config.setdefault("milvus", {})
    search = config.setdefault("search", {})
    paths = config.setdefault("paths", {})
    app = config.setdefault("app", {})

    # -------------------------
    # Model
    # -------------------------

    model["name"] = os.getenv(
        "CLIP_MODEL_NAME",
        model.get("name", "openai/clip-vit-base-patch32"),
    )

    model["device"] = os.getenv(
        "CLIP_DEVICE",
        model.get("device", "auto"),
    )

    model["embedding_dim"] = int(
        os.getenv(
            "EMBEDDING_DIM",
            model.get("embedding_dim", 512),
        )
    )

    # -------------------------
    # Milvus
    # -------------------------

    milvus["mode"] = os.getenv(
        "MILVUS_MODE",
        milvus.get("mode", "lite"),
    ).lower()

    milvus["uri"] = os.getenv(
        "MILVUS_URI",
        milvus.get("uri", "data/milvus/aic2026_milvus.db"),
    )

    milvus["server_uri"] = os.getenv(
        "MILVUS_SERVER_URI",
        milvus.get("server_uri", "http://localhost:19530"),
    )

    milvus["token"] = os.getenv(
        "MILVUS_TOKEN",
        milvus.get("token", ""),
    )

    milvus["collection_name"] = os.getenv(
        "COLLECTION_NAME",
        milvus.get("collection_name", "aic2026_keyframes"),
    )

    milvus["metric_type"] = os.getenv(
        "MILVUS_METRIC",
        milvus.get("metric_type", "IP"),
    )

    milvus["nprobe"] = int(
        os.getenv(
            "MILVUS_NPROBE",
            milvus.get("nprobe", 16),
        )
    )

    # -------------------------
    # Search
    # -------------------------

    search["top_k"] = int(
        os.getenv(
            "TOP_K",
            search.get("top_k", 100),
        )
    )

    search["query_expansion"] = _env_bool(
        "QUERY_EXPANSION",
        search.get("query_expansion", True),
    )

    # -------------------------
    # App
    # -------------------------

    app["use_mock"] = _env_bool(
        "USE_MOCK",
        app.get("use_mock", False),
    )

    app["host"] = os.getenv(
        "APP_HOST",
        app.get("host", "127.0.0.1"),
    )

    app["port"] = int(
        os.getenv(
            "APP_PORT",
            app.get("port", 7860),
        )
    )

    # -------------------------
    # Resolve project paths
    # -------------------------

    for key in ("manifest_keyframes", "manifest_videos"):
        if key in paths:
            paths[key] = str(resolve_project_path(paths[key]))

    roots = paths.setdefault("part_roots", {})

    for part, root in roots.items():
        roots[part] = str(resolve_project_path(root))

    # -------------------------
    # Milvus URI
    # -------------------------

    if milvus["mode"] == "lite":
        milvus["resolved_uri"] = str(
            resolve_project_path(milvus["uri"])
        )

    elif milvus["mode"] == "server":
        milvus["resolved_uri"] = milvus["server_uri"]

    else:
        raise ValueError(
            "milvus.mode phải là 'lite' hoặc 'server'."
        )

    return config