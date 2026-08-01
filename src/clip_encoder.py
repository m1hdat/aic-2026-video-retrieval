from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class ClipEncoder:
    """Small wrapper around CLIP image and text embedding APIs."""

    def __init__(self, model_name: str, device: str = "auto") -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode_image(self, image_path: str | Path) -> np.ndarray:
        """Encode one image path into a normalized CLIP vector."""
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            features = self.model.get_image_features(**inputs)

        features = self._feature_tensor(features)
        return self._to_numpy(features)

    def encode_text(self, text: str) -> np.ndarray:
        """Encode one text query into a normalized CLIP vector."""
        inputs = self.processor(
            text=[text],
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            features = self.model.get_text_features(**inputs)

        features = self._feature_tensor(features)
        return self._to_numpy(features)

    @staticmethod
    def _feature_tensor(features) -> torch.Tensor:
        """Handle CLIP feature outputs across transformers versions."""
        if isinstance(features, torch.Tensor):
            return features

        if hasattr(features, "image_embeds") and features.image_embeds is not None:
            return features.image_embeds

        if hasattr(features, "text_embeds") and features.text_embeds is not None:
            return features.text_embeds

        if hasattr(features, "pooler_output") and features.pooler_output is not None:
            return features.pooler_output

        if isinstance(features, (tuple, list)) and features:
            first = features[0]
            if isinstance(first, torch.Tensor):
                return first

        raise TypeError(f"Unsupported CLIP feature output type: {type(features)!r}")

    @staticmethod
    def _to_numpy(features: torch.Tensor) -> np.ndarray:
        # Normalization makes inner product behave like cosine similarity.
        normalized = functional.normalize(features, p=2, dim=-1)
        return normalized.cpu().numpy().astype("float32")[0]


def load_clip_model(model_name: str, device: str = "auto") -> ClipEncoder:
    return ClipEncoder(model_name=model_name, device=device)


def encode_image(image_path: str | Path, encoder: ClipEncoder) -> np.ndarray:
    return encoder.encode_image(image_path)


def encode_text(text: str, encoder: ClipEncoder) -> np.ndarray:
    return encoder.encode_text(text)
