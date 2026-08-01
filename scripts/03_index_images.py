from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.indexer import index_precomputed_embeddings
from src.milvus_client import connect_milvus, create_collection


def main() -> None:
    config = load_config()
    milvus_config = config["milvus"]
    indexing_config = config.get("indexing", {})

    connect_milvus(host=milvus_config["host"], port=milvus_config["port"])
    collection = create_collection(config)

    inserted_count = index_precomputed_embeddings(
        metadata_path=config["data"]["metadata_path"],
        embeddings_path=config["data"]["embeddings_path"],
        collection=collection,
        batch_size=indexing_config.get("batch_size", 256),
        max_images=indexing_config.get("max_images"),
    )

    print(f"Inserted {inserted_count} precomputed image embeddings into Milvus.")


if __name__ == "__main__":
    main()
