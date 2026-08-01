from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.milvus_client import connect_milvus
from src.search_engine import SearchEngine, format_search_results


def load_default_query() -> str:
    sample_path = PROJECT_ROOT / "data" / "sample_queries.txt"
    if sample_path.exists():
        return sample_path.read_text(encoding="utf-8").splitlines()[0]
    return "a dog running on the grass"


def main() -> None:
    config = load_config()
    milvus_config = config["milvus"]

    connect_milvus(host=milvus_config["host"], port=milvus_config["port"])
    engine = SearchEngine(config)

    query = " ".join(sys.argv[1:]).strip() or load_default_query()
    results = engine.search_by_text(query, top_k=config["search"].get("top_k", 5))

    print(f"Query: {query}\n")
    print(f"Top {len(results)} results:")
    print(format_search_results(results))


if __name__ == "__main__":
    main()

