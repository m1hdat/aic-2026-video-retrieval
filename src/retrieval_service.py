from __future__ import annotations

from pathlib import Path
from typing import Any

from frontend.mock_data import (
    build_mock_kis_results,
    build_mock_qa_results,
    build_mock_trake_results,
)


class RetrievalService:
    """
    Lớp trung gian giữa Gradio và backend retrieval.

    Frontend chỉ gọi ba hàm:
      - search_kis(...)
      - search_qa(...)
      - search_trake(...)

    Khi backend thật chưa sẵn sàng, use_mock=True.
    Khi dataset + Milvus hoàn tất, đặt USE_MOCK=false và giữ nguyên frontend.
    """

    def __init__(
        self,
        project_root: Path,
        use_mock: bool = True,
        mock_asset_dir: Path | None = None,
    ) -> None:
        self.project_root = Path(project_root)
        self.use_mock = use_mock
        self.mock_asset_dir = mock_asset_dir or self.project_root / "assets" / "mock"
        self.engine = None
        self.config = None

        if not self.use_mock:
            self._initialize_real_backend()

    def _initialize_real_backend(self) -> None:
        """
        Tận dụng backend cũ của repo.

        Không kết nối Milvus khi USE_MOCK=true, nhờ vậy frontend vẫn chạy
        trong giai đoạn chưa xử lý dataset.
        """
        try:
            from src.config import load_config
            from src.milvus_client import connect_milvus
            from src.search_engine import SearchEngine

            self.config = load_config()
            connect_milvus(
                host=self.config["milvus"]["host"],
                port=self.config["milvus"]["port"],
            )
            self.engine = SearchEngine(self.config)
        except Exception as exc:
            raise RuntimeError(
                "Không thể khởi tạo backend thật. "
                "Hãy chạy với USE_MOCK=true hoặc kiểm tra Docker/Milvus/config."
            ) from exc

    def search_kis(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        if self.use_mock:
            return build_mock_kis_results(
                query=query,
                top_k=top_k,
                asset_dir=self.mock_asset_dir,
            )

        raw_results = self.engine.search_by_text(query, top_k=top_k)
        return self._adapt_real_results(raw_results)

    def search_qa(
        self,
        event_description: str,
        question: str,
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        if self.use_mock:
            return build_mock_qa_results(
                event_description=event_description,
                question=question,
                top_k=top_k,
                asset_dir=self.mock_asset_dir,
            )

        # Baseline ban đầu: ghép mô tả sự kiện và câu hỏi thành text query.
        # Sau này nhóm model có thể thay phần này bằng VLM/QA service mà
        # không phải chỉnh gradio_app.py.
        query = " ".join(
            part.strip()
            for part in (event_description, question)
            if part and part.strip()
        )
        raw_results = self.engine.search_by_text(query, top_k=top_k)
        return self._adapt_real_results(raw_results)

    def search_trake(
        self,
        events: list[str],
        top_videos: int = 10,
    ) -> list[dict[str, Any]]:
        if self.use_mock:
            return build_mock_trake_results(events, top_videos)

        # Baseline thật tạm thời:
        # 1) search từng event;
        # 2) gom kết quả theo video_id;
        # 3) lấy frame tốt nhất của mỗi event;
        # 4) thưởng video có đủ event và frame tăng dần.
        per_event_results = [
            self._adapt_real_results(
                self.engine.search_by_text(event, top_k=max(100, top_videos * 20))
            )
            for event in events
        ]
        return self._aggregate_trake(events, per_event_results, top_videos)

    def _adapt_real_results(
        self,
        raw_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Chuẩn hóa output backend cũ/mới về schema frontend dùng.

        Backend AIC thật nên trả trực tiếp:
        rank, score, video_id, frame_id, keyframe_id,
        image_path, video_path, objects.
        """
        adapted = []

        for index, item in enumerate(raw_results, start=1):
            image_path = item.get("image_path", "")
            video_id = item.get("video_id") or self._infer_video_id(image_path)
            keyframe_id = item.get("keyframe_id") or Path(image_path).stem
            frame_id = item.get("frame_id")

            if frame_id is None:
                # Chỉ là fallback để frontend không crash.
                # Khi dữ liệu AIC được chuẩn hóa, frame_id phải lấy từ metadata BTC.
                try:
                    frame_id = int(keyframe_id)
                except (TypeError, ValueError):
                    frame_id = 0

            objects = item.get("objects", [])
            if isinstance(objects, str):
                objects = [x.strip() for x in objects.split(",") if x.strip()]

            adapted.append(
                {
                    "rank": int(item.get("rank", index)),
                    "score": float(item.get("score", 0.0)),
                    "video_id": str(video_id),
                    "frame_id": int(frame_id),
                    "keyframe_id": str(keyframe_id),
                    "image_path": str(image_path),
                    "video_path": item.get("video_path"),
                    "objects": objects,
                    "metadata_text": item.get(
                        "metadata_text",
                        item.get("caption", ""),
                    ),
                }
            )

        return adapted

    @staticmethod
    def _infer_video_id(image_path: str) -> str:
        path = Path(image_path)
        if path.parent.name:
            return path.parent.name
        return "UNKNOWN_VIDEO"

    @staticmethod
    def _aggregate_trake(
        events: list[str],
        per_event_results: list[list[dict[str, Any]]],
        top_videos: int,
    ) -> list[dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}

        for event_index, results in enumerate(per_event_results):
            for result in results:
                video_id = result["video_id"]
                bucket = grouped.setdefault(
                    video_id,
                    {
                        "video_id": video_id,
                        "event_hits": {},
                    },
                )

                current = bucket["event_hits"].get(event_index)
                if current is None or result["score"] > current["score"]:
                    bucket["event_hits"][event_index] = result

        candidates = []
        event_count = max(1, len(events))

        for video_id, bucket in grouped.items():
            hits = bucket["event_hits"]
            coverage = len(hits) / event_count
            if not hits:
                continue

            ordered_hits = [hits.get(i) for i in range(event_count)]
            frame_ids = [
                int(hit["frame_id"]) if hit is not None else -1
                for hit in ordered_hits
            ]
            valid_frames = [frame for frame in frame_ids if frame >= 0]
            mean_score = sum(hit["score"] for hit in hits.values()) / len(hits)

            temporal_ok = (
                len(valid_frames) == event_count
                and valid_frames == sorted(valid_frames)
            )
            temporal_factor = 1.0 if temporal_ok else 0.65
            final_score = mean_score * coverage * temporal_factor

            candidates.append(
                {
                    "video_id": video_id,
                    "score": float(final_score),
                    "events": list(events),
                    "frame_ids": frame_ids,
                }
            )

        candidates.sort(key=lambda item: item["score"], reverse=True)

        for rank, item in enumerate(candidates[:top_videos], start=1):
            item["rank"] = rank

        return candidates[:top_videos]
