from __future__ import annotations

from typing import Any

from src.clip_encoder import ClipEncoder
from src.milvus_client import get_collection, search_vectors


class SearchEngine:
    """Online text-to-image search service."""

    def __init__(self, config: dict[str, Any], encoder: ClipEncoder | None = None) -> None:
        self.config = config
        self.encoder = encoder or ClipEncoder(
            model_name=config["model"]["name"],
            device=config["model"].get("device", "auto"),
        )
        self.collection = get_collection(config["milvus"]["collection_name"])

    def search_by_text(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """Encode a text query and retrieve the nearest image vectors."""
        if not query.strip():
            return []

        milvus = self.config["milvus"]
        query_embedding = self.encoder.encode_text(query).tolist()
        return search_vectors(
            collection=self.collection,
            query_embedding=query_embedding,
            top_k=top_k or self.config["search"].get("top_k", 5),
            nprobe=milvus.get("nprobe", 10),
            metric_type=milvus.get("metric_type", "IP"),
        )


def search_by_text(engine: SearchEngine, query: str, top_k: int = 5) -> list[dict[str, Any]]:
    return engine.search_by_text(query, top_k=top_k)


def format_search_results(results: list[dict[str, Any]]) -> str:
    lines = []
    for item in results:
        lines.append(
            f'{item["rank"]}. {item["image_path"]} | score: {item["score"]:.4f} | {item["caption"]}'
        )
    return "\n".join(lines)

