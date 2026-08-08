from __future__ import annotations
import re, torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from .settings import settings

VI_MARKS=re.compile(r'[ăâđêôơưĂÂĐÊÔƠƯ]|[àáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]',re.I)

class Translator:
    def __init__(self): self.tokenizer=None; self.model=None
    def to_english(self,text: str) -> str:
        text=text.strip()
        if not text or not VI_MARKS.search(text): return text
        if self.model is None:
            self.tokenizer=AutoTokenizer.from_pretrained(settings.translation_model)
            self.model=AutoModelForSeq2SeqLM.from_pretrained(settings.translation_model).eval()
        batch=self.tokenizer([text],return_tensors='pt',truncation=True,max_length=256)
        with torch.inference_mode(): out=self.model.generate(**batch,max_new_tokens=256,num_beams=4)
        return self.tokenizer.decode(out[0],skip_special_tokens=True)
