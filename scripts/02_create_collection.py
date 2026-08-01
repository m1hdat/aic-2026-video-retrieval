from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.milvus_client import connect_milvus, create_collection


def main() -> None:
    config = load_config()
    milvus_config = config["milvus"]

    connect_milvus(host=milvus_config["host"], port=milvus_config["port"])
    collection = create_collection(config)

    print(f"Collection is ready: {collection.name}")


if __name__ == "__main__":
    main()

