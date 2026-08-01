from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
MAX_IMAGES = os.getenv("MAX_IMAGES")
MAX_IMAGES = int(MAX_IMAGES) if MAX_IMAGES else None
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/kaggle/working"))
LOCAL_IMAGE_PREFIX = os.getenv("LOCAL_IMAGE_PREFIX", "flickr8k/versions/1/Images")


def first_existing_path(candidates: list[str | Path]) -> Path:
    """Return the first existing path from common Kaggle dataset layouts."""
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return path
    raise FileNotFoundError(f"None of these paths exist: {candidates}")


def find_kaggle_paths() -> tuple[Path, Path]:
    dataset_root = Path(os.getenv("KAGGLE_FLICKR8K_ROOT", "/kaggle/input/flickr8k"))
    image_dir = first_existing_path(
        [
            os.getenv("IMAGE_DIR", ""),
            dataset_root / "Images",
            dataset_root / "images",
            dataset_root / "versions" / "1" / "Images",
        ]
    )
    captions_path = first_existing_path(
        [
            os.getenv("CAPTIONS_PATH", ""),
            dataset_root / "captions.txt",
            dataset_root / "versions" / "1" / "captions.txt",
        ]
    )
    return image_dir, captions_path


def load_captions(captions_path: Path) -> dict[str, list[str]]:
    """Load Flickr8k captions and group them by image filename."""
    captions: dict[str, list[str]] = defaultdict(list)
    with captions_path.open("r", encoding="utf-8", newline="") as file:
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


def list_images(image_dir: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted(
        path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in extensions
    )
    return image_paths[:MAX_IMAGES] if MAX_IMAGES else image_paths


def feature_tensor(features) -> torch.Tensor:
    """Handle CLIP image feature outputs across transformers versions."""
    if isinstance(features, torch.Tensor):
        return features
    if hasattr(features, "image_embeds") and features.image_embeds is not None:
        return features.image_embeds
    if hasattr(features, "pooler_output") and features.pooler_output is not None:
        return features.pooler_output
    if isinstance(features, (tuple, list)) and features and isinstance(features[0], torch.Tensor):
        return features[0]
    raise TypeError(f"Unsupported CLIP feature output type: {type(features)!r}")


def encode_images(image_paths: list[Path], model: CLIPModel, processor: CLIPProcessor, device: str) -> np.ndarray:
    """Encode all images into normalized CLIP vectors."""
    all_embeddings = []

    for start in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="Encoding images"):
        batch_paths = image_paths[start : start + BATCH_SIZE]
        images = [Image.open(path).convert("RGB") for path in batch_paths]
        inputs = processor(images=images, return_tensors="pt").to(device)

        with torch.no_grad():
            features = feature_tensor(model.get_image_features(**inputs))
            features = functional.normalize(features, p=2, dim=-1)

        all_embeddings.append(features.cpu().numpy().astype("float32"))

    return np.vstack(all_embeddings)


def build_metadata(image_paths: list[Path], captions: dict[str, list[str]]) -> pd.DataFrame:
    """Build metadata in the same order as the embedding array."""
    rows = []
    for image_id, image_path in enumerate(image_paths, start=1):
        image_captions = captions.get(image_path.name, [])
        rows.append(
            {
                "image_id": image_id,
                "image_path": f"{LOCAL_IMAGE_PREFIX}/{image_path.name}",
                "caption": image_captions[0] if image_captions else "",
                "all_captions": " | ".join(image_captions),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    image_dir, captions_path = find_kaggle_paths()
    image_paths = list_images(image_dir)
    captions = load_captions(captions_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    metadata = build_metadata(image_paths, captions)
    embeddings = encode_images(image_paths, model, processor, device)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata_path = OUTPUT_DIR / "image_metadata.csv"
    embeddings_path = OUTPUT_DIR / "image_embeddings.npy"

    metadata.to_csv(metadata_path, index=False, encoding="utf-8")
    np.save(embeddings_path, embeddings)

    print(f"Saved metadata: {metadata_path}")
    print(f"Saved embeddings: {embeddings_path}")
    print(f"Embedding shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
