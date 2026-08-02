from __future__ import annotations

import json

from src.config import load_config
from src.dataset_loader import load_manifest, validate_manifest


def main() -> None:
    config = load_config()
    manifest = load_manifest(config["paths"]["manifest_keyframes"])
    checks = validate_manifest(manifest)
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    print(f"Rows: {len(manifest):,}")
    print(f"Videos: {manifest['video_id'].nunique():,}")
    if not all(checks.values()):
        raise SystemExit("Manifest chưa đạt yêu cầu.")


if __name__ == "__main__":
    main()
