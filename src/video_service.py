from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


class VideoService:
    """
    Tạo đoạn preview quanh frame đã chọn.

    Nếu chưa có video thật hoặc chưa cài OpenCV, hàm trả None để Gradio
    vẫn hoạt động bình thường trong mock mode.
    """

    def __init__(
        self,
        project_root: Path,
        temp_dir: Path,
        before_seconds: float = 5.0,
        after_seconds: float = 5.0,
    ) -> None:
        self.project_root = Path(project_root)
        self.temp_dir = Path(temp_dir)
        self.before_seconds = float(before_seconds)
        self.after_seconds = float(after_seconds)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def create_preview(
        self,
        video_path: str | Path | None,
        frame_id: int,
    ) -> str | None:
        resolved = self.resolve_video_path(video_path)
        if resolved is None:
            return None

        try:
            import cv2
        except ImportError:
            return str(resolved)

        capture = cv2.VideoCapture(str(resolved))
        if not capture.isOpened():
            capture.release()
            return str(resolved)

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        if fps <= 0 or width <= 0 or height <= 0:
            capture.release()
            return str(resolved)

        start_frame = max(0, int(frame_id - self.before_seconds * fps))
        end_frame = int(frame_id + self.after_seconds * fps)
        if total_frames > 0:
            end_frame = min(end_frame, total_frames - 1)

        cache_key = hashlib.sha1(
            f"{resolved}|{start_frame}|{end_frame}".encode("utf-8")
        ).hexdigest()[:16]
        output_path = self.temp_dir / f"preview_{cache_key}.mp4"

        if output_path.exists() and output_path.stat().st_size > 0:
            capture.release()
            return str(output_path)

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        if not writer.isOpened():
            capture.release()
            writer.release()
            return str(resolved)

        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current = start_frame

        while current <= end_frame:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
            current += 1

        capture.release()
        writer.release()

        if output_path.exists() and output_path.stat().st_size > 0:
            return str(output_path)
        return str(resolved)

    def resolve_video_path(
        self,
        video_path: str | Path | None,
    ) -> Path | None:
        if not video_path:
            return None

        candidate = Path(video_path)
        if candidate.exists():
            return candidate.resolve()

        project_candidate = self.project_root / candidate
        if project_candidate.exists():
            return project_candidate.resolve()

        return None

    @staticmethod
    def frame_to_timestamp(frame_id: int, fps: float) -> float:
        if fps <= 0:
            raise ValueError("FPS phải lớn hơn 0.")
        return max(0.0, float(frame_id) / float(fps))
