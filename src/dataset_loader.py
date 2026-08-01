from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError

from src.config import PROJECT_ROOT, resolve_project_path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_image_paths(image_dir: str | Path) -> list[Path]:
    """Return readable image paths sorted by filename."""
    root = resolve_project_path(image_dir)
    if not root.exists():
        raise FileNotFoundError(f"Image directory does not exist: {root}")

    return sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_captions(captions_path: str | Path | None) -> dict[str, list[str]]:
    """Load Flickr-style captions and group them by image filename."""
    if not captions_path:
        return {}

    path = resolve_project_path(captions_path)
    if not path.exists():
        return {}

    captions: dict[str, list[str]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames and {"image", "caption"}.issubset(reader.fieldnames):
            for row in reader:
                captions[row["image"]].append(row["caption"])
        else:
            file.seek(0)
            for line in file:
                image_name, _, caption = line.partition(",")
                if image_name and caption:
                    captions[image_name.strip()].append(caption.strip())

    return dict(captions)


def is_valid_image(image_path: str | Path) -> bool:
    """Verify that PIL can open the image without decoding the full file."""
    try:
        with Image.open(image_path) as image:
            image.verify()
        return True
    except (OSError, UnidentifiedImageError):
        return False


def build_metadata(
    image_dir: str | Path,
    captions_path: str | Path | None,
    output_path: str | Path | None = None,
    validate_images: bool = True,
) -> pd.DataFrame:
    """Build one metadata row per image for indexing and search display."""
    captions = load_captions(captions_path)
    rows = []

    for index, image_path in enumerate(get_image_paths(image_dir), start=1):
        if validate_images and not is_valid_image(image_path):
            continue

        image_captions = captions.get(image_path.name, [])
        relative_path = image_path.relative_to(PROJECT_ROOT).as_posix()
        rows.append(
            {
                "image_id": index,
                "image_path": relative_path,
                "caption": image_captions[0] if image_captions else "",
                "all_captions": " | ".join(image_captions),
            }
        )

    metadata = pd.DataFrame(rows)

    if output_path:
        output = resolve_project_path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata.to_csv(output, index=False, encoding="utf-8")

    return metadata

