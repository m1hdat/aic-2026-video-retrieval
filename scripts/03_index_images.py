from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import load_config, resolve_project_path
from src.milvus_client import MilvusManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index BTC CLIP features into Milvus")
    parser.add_argument("--common-root", required=True, help="Root of aic2026-batch1-common")
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def safe_int(value, default=-1) -> int:
    return default if pd.isna(value) else int(value)


def safe_float(value, default=-1.0) -> float:
    return default if pd.isna(value) else float(value)


def main() -> None:
    args = parse_args()
    config = load_config()
    common_root = Path(args.common_root)
    manifest = pd.read_parquet(resolve_project_path(config["paths"]["manifest_keyframes"]))

    manager = MilvusManager(config)
    manager.ensure_collection(drop_existing=args.rebuild)

    pending: list[dict] = []
    inserted = 0
    for relpath, rows in tqdm(
        manifest.groupby("clip_feature_relpath", sort=False),
        total=manifest["clip_feature_relpath"].nunique(),
        desc="Indexing AIC features",
    ):
        feature_path = common_root / str(relpath)
        if not feature_path.exists():
            raise FileNotFoundError(feature_path)
        all_vectors = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        indices = rows["feature_row"].astype(int).to_numpy()
        vectors = np.asarray(all_vectors[indices], dtype=np.float32)
        vectors /= np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)

        for vector, (_, row) in zip(vectors, rows.iterrows()):
            pending.append({
                "id": int(row["global_id"]),
                "embedding": vector.tolist(),
                "video_id": str(row["video_id"]),
                "frame_id": safe_int(row["frame_id"]),
                "source_part": str(row["source_part"]),
                "feature_row": safe_int(row["feature_row"]),
                "keyframe_relpath": str(row.get("keyframe_relpath", "")),
                "video_relpath": str(row.get("video_relpath", "")),
                "timestamp_sec": safe_float(row.get("timestamp_sec")),
            })
            if len(pending) >= args.batch_size:
                manager.insert(pending)
                inserted += len(pending)
                pending.clear()
        del all_vectors, vectors
        gc.collect()

    if pending:
        manager.insert(pending)
        inserted += len(pending)
    print(f"Inserted {inserted:,} entities into {manager.collection_name}")


if __name__ == "__main__":
    main()
