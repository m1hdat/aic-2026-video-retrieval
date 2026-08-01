from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


class SubmissionManager:
    MAX_ANSWERS = 100

    def __init__(self, submission_dir: Path, session_dir: Path) -> None:
        self.submission_dir = Path(submission_dir)
        self.session_dir = Path(session_dir)
        self.submission_dir.mkdir(parents=True, exist_ok=True)
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def add(
        self,
        mode: str,
        queue: list[dict[str, Any]] | None,
        answer: dict[str, Any],
    ) -> list[dict[str, Any]]:
        queue = [dict(item) for item in (queue or [])]

        if len(queue) >= self.MAX_ANSWERS:
            raise ValueError("Mỗi truy vấn chỉ được giữ tối đa 100 đáp án.")

        key = self._answer_key(mode, answer)
        if any(self._answer_key(mode, item) == key for item in queue):
            raise ValueError("Đáp án này đã tồn tại trong queue.")

        queue.append(dict(answer))
        return self._rerank(queue)

    def delete(
        self,
        queue: list[dict[str, Any]] | None,
        rank: Any,
    ) -> list[dict[str, Any]]:
        queue = [dict(item) for item in (queue or [])]
        index = self._rank_to_index(rank, len(queue))
        if index is None:
            return self._rerank(queue)

        queue.pop(index)
        return self._rerank(queue)

    def move(
        self,
        queue: list[dict[str, Any]] | None,
        rank: Any,
        direction: int,
    ) -> list[dict[str, Any]]:
        queue = [dict(item) for item in (queue or [])]
        index = self._rank_to_index(rank, len(queue))
        if index is None:
            return self._rerank(queue)

        target = index + direction
        if 0 <= target < len(queue):
            queue[index], queue[target] = queue[target], queue[index]

        return self._rerank(queue)

    def to_dataframe(
        self,
        mode: str,
        queue: list[dict[str, Any]] | None,
    ) -> pd.DataFrame:
        queue = self._rerank([dict(item) for item in (queue or [])])

        if mode == "kis":
            columns = ["rank", "video_id", "frame_id", "score"]
        elif mode == "qa":
            columns = ["rank", "video_id", "frame_id", "answer", "score"]
        elif mode == "trake":
            columns = ["rank", "video_id", "frame_ids", "score"]
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        rows = []
        for item in queue:
            row = dict(item)
            if mode == "trake":
                row["frame_ids"] = ", ".join(
                    str(value) for value in row.get("frame_ids", [])
                )
            rows.append([row.get(column, "") for column in columns])

        return pd.DataFrame(rows, columns=columns)

    def export_csv(
        self,
        mode: str,
        queue: list[dict[str, Any]],
    ) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.submission_dir / f"{mode}_submission_{timestamp}.csv"
        dataframe = self.to_dataframe(mode, queue)

        # score chỉ phục vụ xếp hạng nội bộ, không phải trường đáp án chính thức.
        if "score" in dataframe.columns:
            dataframe = dataframe.drop(columns=["score"])

        dataframe.to_csv(path, index=False, encoding="utf-8-sig")
        return str(path)

    def save_session(
        self,
        mode: str,
        queue: list[dict[str, Any]],
    ) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.session_dir / f"{mode}_session_{timestamp}.json"
        payload = {
            "mode": mode,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "answers": self._rerank([dict(item) for item in queue]),
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(path)

    def load_session(
        self,
        uploaded_file: Any,
        expected_mode: str,
    ) -> list[dict[str, Any]]:
        if uploaded_file is None:
            raise ValueError("Chưa chọn file session.")

        path = self._extract_uploaded_path(uploaded_file)
        payload = json.loads(path.read_text(encoding="utf-8"))

        if payload.get("mode") != expected_mode:
            raise ValueError(
                f"Session thuộc mode {payload.get('mode')!r}, "
                f"không phải {expected_mode!r}."
            )

        answers = payload.get("answers", [])
        if not isinstance(answers, list):
            raise ValueError("Session JSON không hợp lệ.")

        return self._rerank([dict(item) for item in answers[: self.MAX_ANSWERS]])

    @staticmethod
    def parse_frame_ids(text: str) -> list[int]:
        values = re.findall(r"-?\d+", text or "")
        return [int(value) for value in values if int(value) >= 0]

    @staticmethod
    def _extract_uploaded_path(uploaded_file: Any) -> Path:
        if isinstance(uploaded_file, str):
            return Path(uploaded_file)

        name = getattr(uploaded_file, "name", None)
        if name:
            return Path(name)

        raise ValueError("Không đọc được đường dẫn file upload.")

    @staticmethod
    def _rerank(queue: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for index, item in enumerate(queue, start=1):
            item["rank"] = index
        return queue

    @staticmethod
    def _rank_to_index(rank: Any, queue_length: int) -> int | None:
        try:
            index = int(rank) - 1
        except (TypeError, ValueError):
            return None

        if not 0 <= index < queue_length:
            return None
        return index

    @staticmethod
    def _answer_key(mode: str, answer: dict[str, Any]) -> tuple[Any, ...]:
        if mode == "kis":
            return answer.get("video_id"), int(answer.get("frame_id", -1))
        if mode == "qa":
            return (
                answer.get("video_id"),
                int(answer.get("frame_id", -1)),
                str(answer.get("answer", "")).strip().casefold(),
            )
        if mode == "trake":
            return (
                answer.get("video_id"),
                tuple(int(x) for x in answer.get("frame_ids", [])),
            )
        raise ValueError(f"Unsupported mode: {mode}")
