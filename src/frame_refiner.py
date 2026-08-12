from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
from PIL import Image

from .clip_encoder import TextEncoder
from .settings import ROOT, settings
from .video_resolver import resolve_video


class FrameRefiner:
    def __init__(self, encoder: TextEncoder):
        self.encoder = encoder

    @staticmethod
    @lru_cache(maxsize=2048)
    def _resolve_video_cached(video_id: str):
        """Resolve each video path once instead of scanning roots per result."""
        return resolve_video(video_id)

    @lru_cache(maxsize=128)
    def _encode_query_cached(self, query: str) -> np.ndarray:
        """Encode the same translated query only once."""
        return np.asarray(self.encoder.encode([query])[0], dtype=np.float32)

    def _score_images(
        self,
        images: list[Image.Image],
        query_vector: np.ndarray,
    ) -> np.ndarray:
        if not images:
            return np.empty(0, dtype=np.float32)

        batch_size = max(1, settings.refine_batch_size)
        score_batches: list[np.ndarray] = []

        for start in range(0, len(images), batch_size):
            vectors = self.encoder.encode_images(images[start : start + batch_size])
            scores = np.asarray(vectors @ query_vector, dtype=np.float32).reshape(-1)
            score_batches.append(scores)

        return np.concatenate(score_batches)

    @staticmethod
    def _fallback(coarse_frame: int, **extra) -> dict:
        return {
            "frame_idx": coarse_frame,
            "refined": False,
            **extra,
        }

    def refine(
        self,
        video_id: str,
        coarse_frame: int,
        fps_hint: float,
        query: str,
    ) -> dict:
        """Refine a 1-FPS hit with sparse scoring followed by a local fine pass."""
        coarse_frame = max(0, int(coarse_frame))

        if not settings.enable_frame_refine:
            return self._fallback(coarse_frame)

        video_path = self._resolve_video_cached(video_id)
        if not video_path:
            return self._fallback(
                coarse_frame,
                warning="Không tìm thấy video gốc",
            )

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            return self._fallback(
                coarse_frame,
                warning="Không mở được video gốc",
            )

        fps = float(cap.get(cv2.CAP_PROP_FPS) or fps_hint or 25.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        radius = max(1, int(settings.refine_seconds * fps))
        window_start = max(0, coarse_frame - radius)
        window_end = coarse_frame + radius

        if total_frames > 0:
            window_end = min(window_end, total_frames - 1)

        # Seek once, then decode the small window sequentially. Repeated random
        # seeks are particularly expensive for compressed video.
        cap.set(cv2.CAP_PROP_POS_FRAMES, window_start)
        frames_by_id: dict[int, Image.Image] = {}
        frame_idx = window_start

        while frame_idx <= window_end:
            ok, bgr = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames_by_id[frame_idx] = Image.fromarray(rgb)
            frame_idx += 1

        cap.release()

        if not frames_by_id:
            return self._fallback(coarse_frame)

        available_ids = sorted(frames_by_id)
        original_frame = (
            coarse_frame
            if coarse_frame in frames_by_id
            else min(available_ids, key=lambda value: abs(value - coarse_frame))
        )

        query_vector = self._encode_query_cached(query)
        stride = max(1, settings.refine_stride)

        # Stage 1: score a sparse sample across the whole temporal window.
        coarse_ids = available_ids[::stride]
        if original_frame not in coarse_ids:
            coarse_ids.append(original_frame)
        coarse_ids = sorted(set(coarse_ids))
        coarse_images = [frames_by_id[idx] for idx in coarse_ids]
        coarse_scores = self._score_images(coarse_images, query_vector)

        if coarse_scores.size == 0:
            return self._fallback(coarse_frame)

        coarse_best_pos = int(np.argmax(coarse_scores))
        coarse_best_frame = coarse_ids[coarse_best_pos]
        original_pos = coarse_ids.index(original_frame)
        original_score = float(coarse_scores[original_pos])

        # Stage 2: score every native frame only around the best sparse hit.
        fine_start = max(available_ids[0], coarse_best_frame - stride)
        fine_end = min(available_ids[-1], coarse_best_frame + stride)
        fine_ids = [
            idx
            for idx in range(fine_start, fine_end + 1)
            if idx in frames_by_id
        ]
        fine_images = [frames_by_id[idx] for idx in fine_ids]
        fine_scores = self._score_images(fine_images, query_vector)

        if fine_scores.size == 0:
            best_frame = coarse_best_frame
            best_score = float(coarse_scores[coarse_best_pos])
        else:
            fine_best_pos = int(np.argmax(fine_scores))
            best_frame = fine_ids[fine_best_pos]
            best_score = float(fine_scores[fine_best_pos])

        # Keep the original hit unless the new native frame is meaningfully
        # better. This prevents jitter caused by tiny floating-point changes.
        if best_score < original_score + settings.refine_min_gain:
            best_frame = original_frame
            best_score = original_score

        best_image = frames_by_id[best_frame]
        output_path = (
            ROOT
            / "data"
            / "cache"
            / "refined"
            / video_id
            / f"{best_frame:09d}.jpg"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        best_image.save(output_path, format="JPEG", quality=88, optimize=False)

        return {
            "frame_idx": best_frame,
            "refined": best_frame != coarse_frame,
            "refine_score": best_score,
            "refine_original_score": original_score,
            "refined_image_path": str(output_path),
        }