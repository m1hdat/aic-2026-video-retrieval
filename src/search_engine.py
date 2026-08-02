from __future__ import annotations

from typing import Any

from src.clip_encoder import ClipEncoder
from src.metadata_store import MetadataStore
from src.milvus_client import MilvusManager
from src.path_resolver import PathResolver
from src.query_expansion import QueryExpander


class SearchEngine:
    def __init__(
        self,
        config: dict[str, Any],
        encoder: ClipEncoder | None = None,
        milvus: MilvusManager | None = None,
    ) -> None:
        self.config = config
        self.encoder = encoder or ClipEncoder(
            model_name=config["model"]["name"],
            device=config["model"].get("device", "auto"),
        )
        self.milvus = milvus or MilvusManager(config)
        if not self.milvus.collection_exists():
            raise RuntimeError(
                f"Collection '{self.milvus.collection_name}' chưa tồn tại. "
                "Hãy build/index Milvus trước khi chạy web."
            )

        search_cfg = config.get("search", {})
        self.expander = QueryExpander(
            templates=search_cfg.get("expansion_templates"),
            max_variants=search_cfg.get("max_variants", 3),
        )
        self.metadata = MetadataStore(config.get("paths", {}).get("manifest_keyframes"))
        self.paths = PathResolver(config.get("paths", {}).get("part_roots", {}))

    def search_by_text(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        cleaned = " ".join((query or "").split())
        if not cleaned:
            return []

        search_cfg = self.config.get("search", {})
        limit = int(top_k or search_cfg.get("top_k", 100))
        variants = self.expander.expand(cleaned) if search_cfg.get("query_expansion", True) else []
        if not variants:
            variants = self.expander.expand(cleaned)[:1]

        per_variant = max(limit, int(search_cfg.get("per_variant_top_k", limit)))
        fused: dict[int, dict[str, Any]] = {}

        for variant in variants:
            vector = self.encoder.encode_text(variant.text)
            for hit in self.milvus.search(vector, top_k=per_variant):
                item = fused.setdefault(hit["id"], {**hit, "fused_score": 0.0, "matched_queries": []})
                item["fused_score"] += variant.weight * float(hit["score"])
                item["score"] = max(float(item.get("score", -1e9)), float(hit["score"]))
                item["matched_queries"].append(variant.text)

        results = sorted(fused.values(), key=lambda x: x["fused_score"], reverse=True)[:limit]
        for rank, item in enumerate(results, start=1):
            item["rank"] = rank
            item["score"] = float(item["fused_score"] / max(1, len(variants)))
            item = self.metadata.enrich(item)
            source_part = item.get("source_part")
            item["image_path"] = self.paths.resolve(
                source_part,
                item.get("keyframe_relpath"),
                item.get("keyframe_path"),
            )
            item["video_path"] = self.paths.resolve(
                source_part,
                item.get("video_relpath"),
                item.get("video_path"),
            )
            item["keyframe_id"] = str(item.get("keyframe_id") or str(item.get("keyframe_relpath", "")).split("/")[-1].split(".")[0])
            results[rank - 1] = item
        return results


def format_search_results(results: list[dict[str, Any]]) -> str:
    return "\n".join(
        f'{item["rank"]}. {item.get("video_id")} frame={item.get("frame_id")} score={item.get("score", 0):.4f}'
        for item in results
    )
