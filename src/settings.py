from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

@dataclass(frozen=True)
class Settings:
    pg_dsn: str = (f"host={os.getenv('POSTGRES_HOST','localhost')} "
                   f"port={os.getenv('POSTGRES_PORT','5432')} "
                   f"dbname={os.getenv('POSTGRES_DB','aic2026')} "
                   f"user={os.getenv('POSTGRES_USER','aic')} "
                   f"password={os.getenv('POSTGRES_PASSWORD','aic2026')}")
    milvus_uri: str = os.getenv("MILVUS_URI", "http://localhost:19530")
    collection: str = os.getenv("MILVUS_COLLECTION", "aic2026_keyframes")
    clip_model: str = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
    host: str = os.getenv("APP_HOST", "127.0.0.1")
    port: int = int(os.getenv("APP_PORT", "7860"))
    cache_gb: float = float(os.getenv("IMAGE_CACHE_GB", "2"))
    keyframe_roots: str = os.getenv("KEYFRAME_ROOTS", "")
    video_roots: str = os.getenv("VIDEO_ROOTS", os.getenv("VIDEO_SOURCES", ""))
    refine_seconds: float = float(os.getenv("REFINE_SECONDS", "1.5"))
    refine_stride: int = int(os.getenv("REFINE_STRIDE", "1"))
    qa_model: str = os.getenv("QA_MODEL", "Salesforce/blip-vqa-base")
    translation_model: str = os.getenv("TRANSLATION_MODEL", "Helsinki-NLP/opus-mt-vi-en")
    enable_frame_refine: bool = os.getenv("ENABLE_FRAME_REFINE", "true").lower() == "true"
    refine_top_n: int = int(os.getenv("REFINE_TOP_N", "10"))
    trake_refine_top_n: int = int(os.getenv("TRAKE_REFINE_TOP_N", "5"))

settings = Settings()
