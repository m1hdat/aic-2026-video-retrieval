from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


MOCK_OBJECTS = [
    ["person", "laptop"],
    ["person", "microphone", "tree"],
    ["car", "road", "person"],
    ["table", "glass", "person"],
    ["sports equipment", "person"],
    ["screen", "text", "person"],
]


def ensure_mock_assets(asset_dir: Path, count: int = 30) -> list[Path]:
    """Tạo ảnh placeholder để frontend chạy độc lập khi chưa có dataset."""
    asset_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    for index in range(count):
        path = asset_dir / f"mock_{index:03d}.jpg"
        if not path.exists():
            image = Image.new("RGB", (640, 360), (235, 235, 235))
            draw = ImageDraw.Draw(image)
            title = f"AIC MOCK KEYFRAME {index:03d}"
            body = (
                f"video: L{index // 10 + 1:02d}_V{index % 10 + 1:03d}\n"
                f"frame: {(index + 1) * 125}\n"
                "Replace with real keyframe later"
            )
            draw.rectangle((25, 25, 615, 335), outline=(70, 70, 70), width=3)
            draw.text((50, 65), title, fill=(20, 20, 20))
            draw.multiline_text((50, 135), body, fill=(40, 40, 40), spacing=12)
            image.save(path, quality=90)
        created.append(path)

    return created


def _seed_from_text(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def build_mock_kis_results(
    query: str,
    top_k: int,
    asset_dir: Path,
) -> list[dict[str, Any]]:
    assets = ensure_mock_assets(asset_dir)
    rng = random.Random(_seed_from_text(query))
    indices = list(range(len(assets)))
    rng.shuffle(indices)

    results = []
    for rank, image_index in enumerate(indices[:top_k], start=1):
        video_number = image_index % 10 + 1
        batch_number = image_index // 10 + 1
        score = max(0.05, 0.98 - (rank - 1) * 0.018 - rng.random() * 0.015)
        frame_id = (image_index + 1) * 125

        results.append(
            {
                "rank": rank,
                "score": float(score),
                "video_id": f"L{batch_number:02d}_V{video_number:03d}",
                "frame_id": frame_id,
                "keyframe_id": f"{image_index:04d}",
                "image_path": str(assets[image_index]),
                "video_path": None,
                "objects": MOCK_OBJECTS[image_index % len(MOCK_OBJECTS)],
                "metadata_text": f"Mock result for query: {query}",
            }
        )

    return results


def build_mock_qa_results(
    event_description: str,
    question: str,
    top_k: int,
    asset_dir: Path,
) -> list[dict[str, Any]]:
    combined_query = f"{event_description} | {question}"
    return build_mock_kis_results(combined_query, top_k, asset_dir)


def build_mock_trake_results(
    events: list[str],
    top_videos: int,
) -> list[dict[str, Any]]:
    seed = _seed_from_text(" | ".join(events))
    rng = random.Random(seed)
    candidates = []

    for rank in range(1, top_videos + 1):
        video_id = f"L{(rank - 1) // 10 + 1:02d}_V{(rank - 1) % 10 + 1:03d}"
        start = 100 + rng.randint(0, 500) + rank * 15
        frame_ids = []
        current = start

        for _ in events:
            current += rng.randint(35, 140)
            frame_ids.append(current)

        score = max(0.05, 0.95 - (rank - 1) * 0.04 - rng.random() * 0.02)
        candidates.append(
            {
                "rank": rank,
                "video_id": video_id,
                "score": float(score),
                "events": list(events),
                "frame_ids": frame_ids,
            }
        )

    return candidates
