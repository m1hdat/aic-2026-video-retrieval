from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from .settings import settings

class TextEncoder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = CLIPProcessor.from_pretrained(settings.clip_model)
        self.model = CLIPModel.from_pretrained(settings.clip_model).to(self.device).eval()

    def encode(self, texts: list[str]) -> np.ndarray:
        batch = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.inference_mode():
            output = self.model.get_text_features(**batch)
            if not isinstance(output, torch.Tensor):
                text_out = self.model.text_model(**batch, return_dict=True)
                output = self.model.text_projection(text_out.pooler_output)
            output = F.normalize(output, p=2, dim=-1)
        return output.cpu().numpy().astype("float32")

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        batch = self.processor(images=images, return_tensors="pt")
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.inference_mode():
            output = self.model.get_image_features(**batch)
            if not isinstance(output, torch.Tensor):
                vision_out = self.model.vision_model(**batch, return_dict=True)
                output = self.model.visual_projection(vision_out.pooler_output)
            output = F.normalize(output, p=2, dim=-1)
        return output.cpu().numpy().astype("float32")
