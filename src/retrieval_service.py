from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from web.mock_data import build_mock_kis_results, build_mock_qa_results, build_mock_trake_results


class RetrievalService:
    """Application service used by all Gradio tabs."""

    def __init__(
        self,
        project_root: Path,
        use_mock: bool = False,
        mock_asset_dir: Path | None = None,
    ) -> None:
        self.project_root = Path(project_root)
        self.use_mock = bool(use_mock)
        self.mock_asset_dir = mock_asset_dir or self.project_root / "assets" / "mock"
        self.engine = None
        self.config = None
        if not self.use_mock:
            self._initialize_real_backend()

    def _initialize_real_backend(self) -> None:
        try:
            from src.config import load_config
            from src.search_engine import SearchEngine

            self.config = load_config()
            self.engine = SearchEngine(self.config)
        except Exception as exc:
            raise RuntimeError(
                "Không thể khởi tạo backend AIC. Kiểm tra config, CLIP model, "
                "Milvus database/server và collection aic2026_keyframes."
            ) from exc

    def search_kis(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        if self.use_mock:
            return build_mock_kis_results(query=query, top_k=top_k, asset_dir=self.mock_asset_dir)
        return self._adapt_results(self.engine.search_by_text(query, top_k=top_k))

    def search_qa(self, event_description: str, question: str, top_k: int = 20) -> list[dict[str, Any]]:
        if self.use_mock:
            return build_mock_qa_results(
                event_description=event_description,
                question=question,
                top_k=top_k,
                asset_dir=self.mock_asset_dir,
            )

        # Human-in-the-loop Q&A: retrieve the evidence moment; user confirms the answer.
        query = " ".join(part.strip() for part in (event_description, question) if part and part.strip())
        return self._adapt_results(self.engine.search_by_text(query, top_k=top_k))

    def search_trake(self, events: list[str], top_videos: int = 10) -> list[dict[str, Any]]:
        cleaned = [" ".join(event.split()) for event in events if event and event.strip()]
        if self.use_mock:
            return build_mock_trake_results(cleaned, top_videos)
        if len(cleaned) < 2:
            return []

        # Keep several frame candidates per event/video, then solve ordered alignment.
        per_event = [
            self._adapt_results(self.engine.search_by_text(event, top_k=max(300, top_videos * 60)))
            for event in cleaned
        ]
        return self._align_event_sequences(cleaned, per_event, top_videos)

    @staticmethod
    def _adapt_results(raw_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        adapted: list[dict[str, Any]] = []
        for index, item in enumerate(raw_results, start=1):
            objects = item.get("objects", [])
            if isinstance(objects, str):
                objects = [part.strip() for part in objects.split(",") if part.strip()]
            adapted.append({
                **item,
                "rank": int(item.get("rank", index)),
                "score": float(item.get("score", 0.0)),
                "video_id": str(item.get("video_id", "UNKNOWN_VIDEO")),
                "frame_id": int(item.get("frame_id", -1)),
                "keyframe_id": str(item.get("keyframe_id", "")),
                "image_path": item.get("image_path"),
                "video_path": item.get("video_path"),
                "timestamp_sec": float(item.get("timestamp_sec", -1.0) or -1.0),
                "objects": objects,
                "metadata_text": item.get("metadata_text") or item.get("title") or "",
            })
        return adapted

    @staticmethod
    def _align_event_sequences(
        events: list[str],
        per_event_results: list[list[dict[str, Any]]],
        top_videos: int,
    ) -> list[dict[str, Any]]:
        # candidates[event_index][video_id] -> top frames
        by_event: list[dict[str, list[dict[str, Any]]]] = []
        video_coverage: dict[str, int] = defaultdict(int)
        video_best_score: dict[str, float] = defaultdict(float)

        for results in per_event_results:
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for hit in results:
                if int(hit.get("frame_id", -1)) < 0:
                    continue
                grouped[hit["video_id"]].append(hit)
            for video_id, hits in grouped.items():
                hits.sort(key=lambda item: item["score"], reverse=True)
                grouped[video_id] = hits[:40]
                video_coverage[video_id] += 1
                video_best_score[video_id] += hits[0]["score"]
            by_event.append(grouped)

        candidate_videos = sorted(
            video_coverage,
            key=lambda vid: (video_coverage[vid], video_best_score[vid]),
            reverse=True,
        )[: max(100, top_videos * 10)]

        aligned: list[dict[str, Any]] = []
        event_count = len(events)

        for video_id in candidate_videos:
            if video_coverage[video_id] < event_count:
                continue

            # Dynamic programming over ordered frame candidates.
            first = sorted(by_event[0].get(video_id, []), key=lambda x: x["frame_id"])
            states = [([hit], float(hit["score"])) for hit in first]
            if not states:
                continue

            for event_index in range(1, event_count):
                current = sorted(by_event[event_index].get(video_id, []), key=lambda x: x["frame_id"])
                next_states: list[tuple[list[dict[str, Any]], float]] = []
                for hit in current:
                    best_state = None
                    for sequence, score in states:
                        if sequence[-1]["frame_id"] < hit["frame_id"]:
                            candidate_score = score + float(hit["score"])
                            if best_state is None or candidate_score > best_state[1]:
                                best_state = (sequence + [hit], candidate_score)
                    if best_state:
                        next_states.append(best_state)
                states = sorted(next_states, key=lambda state: state[1], reverse=True)[:200]
                if not states:
                    break

            if not states:
                continue
            sequence, score_sum = max(states, key=lambda state: state[1])
            if len(sequence) != event_count:
                continue

            aligned.append({
                "video_id": video_id,
                "score": float(score_sum / event_count),
                "events": events,
                "frame_ids": [int(hit["frame_id"]) for hit in sequence],
                "timestamps_sec": [float(hit.get("timestamp_sec", -1.0)) for hit in sequence],
                "event_hits": sequence,
            })

        aligned.sort(key=lambda item: item["score"], reverse=True)
        for rank, item in enumerate(aligned[:top_videos], start=1):
            item["rank"] = rank
        return aligned[:top_videos]
