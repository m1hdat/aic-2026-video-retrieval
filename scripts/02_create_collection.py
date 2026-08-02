from __future__ import annotations

from src.config import load_config
from src.milvus_client import MilvusManager


def main() -> None:
    config = load_config()
    manager = MilvusManager(config)
    manager.ensure_collection(drop_existing=False)
    print(f"Collection ready: {manager.collection_name}")
    print(f"Mode: {config['milvus']['mode']} | URI: {config['milvus']['resolved_uri']}")


if __name__ == "__main__":
    main()
