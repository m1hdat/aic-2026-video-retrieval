from __future__ import annotations

import argparse

from src.config import load_config
from src.milvus_client import MilvusManager


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true", help="Confirm destructive reset")
    args = parser.parse_args()
    if not args.yes:
        raise SystemExit("Dùng --yes để xác nhận xóa collection.")

    manager = MilvusManager(load_config())
    manager.drop()
    manager.ensure_collection(drop_existing=False)
    print(f"Reset collection: {manager.collection_name}")


if __name__ == "__main__":
    main()
