from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@dataclass(frozen=True)
class Settings:
    pg_dsn: str = (
        f"host={os.getenv('POSTGRES_HOST', 'localhost')} "
        f"port={os.getenv('POSTGRES_PORT', '5432')} "
        f"dbname={os.getenv('POSTGRES_DB', 'aic2026')} "
        f"user={os.getenv('POSTGRES_USER', 'aic')} "
        f"password={os.getenv('POSTGRES_PASSWORD', 'aic2026')}"
    )
    milvus_uri: str = os.getenv("MILVUS_URI", "http://localhost:19530")
    collection: str = os.getenv("MILVUS_COLLECTION", "aic2026_siglip2_keyframes")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "google/siglip2-base-patch16-224")
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "768"))
    host: str = os.getenv("APP_HOST", "127.0.0.1")
    port: int = int(os.getenv("APP_PORT", "7860"))
    cache_gb: float = float(os.getenv("IMAGE_CACHE_GB", "2"))
    keyframe_roots: str = os.getenv("KEYFRAME_ROOTS", "")
    video_roots: str = os.getenv("VIDEO_ROOTS", os.getenv("VIDEO_SOURCES", ""))

    # Frame refinement defaults are intentionally conservative for CPU use.
    refine_seconds: float = float(os.getenv("REFINE_SECONDS", "1.0"))
    refine_stride: int = int(os.getenv("REFINE_STRIDE", "5"))
    refine_batch_size: int = int(os.getenv("REFINE_BATCH_SIZE", "32"))
    refine_min_gain: float = float(os.getenv("REFINE_MIN_GAIN", "0.002"))
    refine_top_n: int = int(os.getenv("REFINE_TOP_N", "3"))
    trake_refine_top_n: int = int(os.getenv("TRAKE_REFINE_TOP_N", "3"))

    qa_model: str = os.getenv("QA_MODEL", "Salesforce/blip-vqa-base")
    google_translate_api_key: str = os.getenv("GOOGLE_TRANSLATE_API_KEY", "")
    google_translate_timeout: float = float(os.getenv("GOOGLE_TRANSLATE_TIMEOUT", "10"))
    enable_google_translate: bool = _env_bool("ENABLE_GOOGLE_TRANSLATE", True)
    enable_frame_refine: bool = _env_bool("ENABLE_FRAME_REFINE", True)
    enable_ocr_search: bool = _env_bool("ENABLE_OCR_SEARCH", True)
    enable_object_search: bool = _env_bool("ENABLE_OBJECT_SEARCH", True)
    signal_candidate_k: int = int(os.getenv("SIGNAL_CANDIDATE_K", "300"))


settings = Settings()