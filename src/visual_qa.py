from __future__ import annotations
from pathlib import Path
import torch
from PIL import Image
from transformers import BlipForQuestionAnswering, BlipProcessor
from .settings import settings

class VisualQA:
    """Real local VQA inference; model is lazy-loaded only when QA is used."""
    def __init__(self): self.device='cuda' if torch.cuda.is_available() else 'cpu'; self.processor=None; self.model=None
    def _load(self):
        if self.model is None:
            self.processor=BlipProcessor.from_pretrained(settings.qa_model)
            self.model=BlipForQuestionAnswering.from_pretrained(settings.qa_model).to(self.device).eval()
    def answer(self, image_path: str, question: str) -> str:
        self._load(); image=Image.open(image_path).convert('RGB')
        inputs=self.processor(images=image,text=question,return_tensors='pt').to(self.device)
        with torch.inference_mode(): output=self.model.generate(**inputs,max_new_tokens=20,num_beams=5)
        return self.processor.decode(output[0],skip_special_tokens=True).strip()
