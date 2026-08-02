from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.retrieval_service import RetrievalService
from src.submission_manager import SubmissionManager
from src.video_service import VideoService
from web.components import build_kis_tab, build_qa_tab, build_trake_tab
from web.mock_data import ensure_mock_assets

CONFIG = load_config()
MOCK_ASSET_DIR = PROJECT_ROOT / "assets" / "mock"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SUBMISSION_DIR = OUTPUT_DIR / "submissions"
SESSION_DIR = OUTPUT_DIR / "sessions"
TEMP_CLIP_DIR = OUTPUT_DIR / "temp_clips"

for directory in (MOCK_ASSET_DIR, SUBMISSION_DIR, SESSION_DIR, TEMP_CLIP_DIR):
    directory.mkdir(parents=True, exist_ok=True)

USE_MOCK = bool(CONFIG.get("app", {}).get("use_mock", False))
if USE_MOCK:
    ensure_mock_assets(MOCK_ASSET_DIR)

retrieval_service = RetrievalService(
    project_root=PROJECT_ROOT,
    use_mock=USE_MOCK,
    mock_asset_dir=MOCK_ASSET_DIR,
)
submission_manager = SubmissionManager(
    submission_dir=SUBMISSION_DIR,
    session_dir=SESSION_DIR,
)
video_service = VideoService(
    project_root=PROJECT_ROOT,
    temp_dir=TEMP_CLIP_DIR,
)

with gr.Blocks(title="AIC 2026 Retrieval Workspace") as demo:
    backend_name = "MOCK" if USE_MOCK else f"MILVUS {CONFIG['milvus']['mode'].upper()}"
    gr.Markdown(
        f"""
# AIC 2026 Retrieval Workspace

**Backend:** `{backend_name}`  
**Collection:** `{CONFIG['milvus']['collection_name']}`  
**CLIP text encoder:** `{CONFIG['model']['name']}`
"""
    )

    build_kis_tab(retrieval_service, submission_manager, video_service)
    build_qa_tab(retrieval_service, submission_manager, video_service)
    build_trake_tab(retrieval_service, submission_manager)


if __name__ == "__main__":
    demo.launch(
        server_name=CONFIG.get("app", {}).get("host", "127.0.0.1"),
        server_port=int(CONFIG.get("app", {}).get("port", 7860)),
        allowed_paths=[
            str(PROJECT_ROOT / "assets"),
            str(PROJECT_ROOT / "outputs"),
            str(PROJECT_ROOT / "data"),
        ],
    )
