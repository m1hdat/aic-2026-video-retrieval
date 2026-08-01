from __future__ import annotations

from typing import Any

import pandas as pd


RESULT_HEADERS = [
    "rank",
    "score",
    "video_id",
    "frame_id",
    "keyframe_id",
    "image_path",
    "objects",
]


def safe_int(value: Any, default: int = 1) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def normalize_select_index(index: Any) -> int | None:
    if isinstance(index, (tuple, list)):
        index = index[0]

    try:
        return int(index)
    except (TypeError, ValueError):
        return None


def results_to_outputs(
    results: list[dict[str, Any]],
) -> tuple[list[tuple[str, str]], pd.DataFrame]:
    gallery_items: list[tuple[str, str]] = []
    rows: list[list[Any]] = []

    for item in results:
        caption = (
            f'#{item["rank"]} | {item["video_id"]} | '
            f'frame={item["frame_id"]} | score={item["score"]:.4f}'
        )
        gallery_items.append((str(item["image_path"]), caption))

        objects = item.get("objects", [])
        if isinstance(objects, str):
            objects_text = objects
        else:
            objects_text = ", ".join(objects)

        rows.append(
            [
                item["rank"],
                round(float(item["score"]), 4),
                item["video_id"],
                item["frame_id"],
                item.get("keyframe_id", ""),
                str(item["image_path"]),
                objects_text,
            ]
        )

    return gallery_items, pd.DataFrame(rows, columns=RESULT_HEADERS)


def parse_events(events_text: str) -> list[str]:
    events = []

    for line in (events_text or "").splitlines():
        cleaned = line.strip().lstrip("-•").strip()
        if cleaned:
            events.append(cleaned)

    return events
