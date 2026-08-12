from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor
from .settings import settings

class TextEncoder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(settings.embedding_model)
        self.model = AutoModel.from_pretrained(settings.embedding_model).to(self.device).eval()

    @staticmethod
    def _unwrap_features(output) -> torch.Tensor:
        """Transformers may return a tensor or BaseModelOutputWithPooling."""
        if torch.is_tensor(output):
            return output
        pooled = getattr(output, "pooler_output", None)
        if torch.is_tensor(pooled):
            return pooled
        raise TypeError(
            "SigLIP2 feature output không được hỗ trợ: "
            f"{type(output).__name__}"
        )

    @staticmethod
    def _validate_dimension(output: torch.Tensor) -> torch.Tensor:
        if output.ndim != 2 or output.shape[1] != settings.embedding_dim:
            raise RuntimeError(
                "Sai chiều embedding SigLIP2: "
                f"nhận {tuple(output.shape)}, cần [N,{settings.embedding_dim}]"
            )
        return output

    def encode(self, texts: list[str]) -> np.ndarray:
        # SigLIP2 was trained with fixed-length text padding.
        batch = self.processor(
            text=texts,
            return_tensors="pt",
            padding="max_length",
            max_length=64,
            truncation=True,
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.inference_mode():
            output = self._unwrap_features(self.model.get_text_features(**batch))
            output = self._validate_dimension(output)
            output = F.normalize(output, p=2, dim=-1)
        return output.cpu().numpy().astype("float32")

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        batch = self.processor(images=images, return_tensors="pt")
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.inference_mode():
            output = self._unwrap_features(self.model.get_image_features(**batch))
            output = self._validate_dimension(output)
            output = F.normalize(output, p=2, dim=-1)
        return output.cpu().numpy().astype("float32")
