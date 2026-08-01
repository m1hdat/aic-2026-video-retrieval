from __future__ import annotations

import os
import sys
from pathlib import Path

import gradio as gr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from frontend.components import build_kis_tab, build_qa_tab, build_trake_tab
from frontend.mock_data import ensure_mock_assets
from src.retrieval_service import RetrievalService
from src.submission_manager import SubmissionManager
from src.video_service import VideoService


MOCK_ASSET_DIR = PROJECT_ROOT / "assets" / "mock"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

SUBMISSION_DIR = OUTPUT_DIR / "submissions"
SESSION_DIR = OUTPUT_DIR / "sessions"
TEMP_CLIP_DIR = OUTPUT_DIR / "temp_clips"

for directory in (
    MOCK_ASSET_DIR,
    SUBMISSION_DIR,
    SESSION_DIR,
    TEMP_CLIP_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)

ensure_mock_assets(MOCK_ASSET_DIR)

USE_MOCK = os.getenv("USE_MOCK", "true").strip().lower() not in {
    "0",
    "false",
    "no",
}

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
    backend_name = "MOCK" if USE_MOCK else "REAL / MILVUS"

    gr.Markdown(
        f"""
# AIC 2026 Retrieval Workspace

**Backend hiện tại:** `{backend_name}`

- `USE_MOCK=true`: chạy giao diện khi chưa có dataset/Milvus.
- `USE_MOCK=false`: dùng backend thật thông qua `RetrievalService`.
"""
    )

    build_kis_tab(
        retrieval_service=retrieval_service,
        submission_manager=submission_manager,
        video_service=video_service,
    )

    build_qa_tab(
        retrieval_service=retrieval_service,
        submission_manager=submission_manager,
        video_service=video_service,
    )

    build_trake_tab(
        retrieval_service=retrieval_service,
        submission_manager=submission_manager,
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        allowed_paths=[
            str(PROJECT_ROOT / "assets"),
            str(PROJECT_ROOT / "outputs"),
            str(PROJECT_ROOT / "data"),
        ],
    )
