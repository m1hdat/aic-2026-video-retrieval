"""BTC already provides CLIP image features; validate them instead of re-encoding."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--common-root", required=True)
    args = parser.parse_args()

    manifest = pd.read_parquet(args.manifest)
    root = Path(args.common_root)
    total = 0
    for relpath, rows in tqdm(manifest.groupby("clip_feature_relpath"), desc="Validating BTC features"):
        path = root / str(relpath)
        vectors = np.load(path, mmap_mode="r", allow_pickle=False)
        if vectors.ndim != 2 or vectors.shape[1] != 512:
            raise RuntimeError(f"Invalid feature shape: {path} -> {vectors.shape}")
        if int(rows["feature_row"].max()) >= vectors.shape[0]:
            raise RuntimeError(f"feature_row out of range: {path}")
        total += len(rows)
    print(f"Validated {total:,} manifest rows. No image re-encoding is required.")


if __name__ == "__main__":
    main()
