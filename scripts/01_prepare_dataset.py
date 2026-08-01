from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.dataset_loader import build_metadata


def main() -> None:
    config = load_config()
    data_config = config["data"]

    metadata = build_metadata(
        image_dir=data_config["image_dir"],
        captions_path=data_config["captions_path"],
        output_path=data_config["metadata_path"],
        validate_images=True,
    )

    print(f"Created metadata with {len(metadata)} images: {data_config['metadata_path']}")


if __name__ == "__main__":
    main()

