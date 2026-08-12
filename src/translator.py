from __future__ import annotations
import html
import json
from functools import lru_cache
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .settings import settings

class Translator:
    """Google Cloud Translation Basic v2; no local translation model is loaded."""

    @lru_cache(maxsize=2048)
    def to_english(self,text: str) -> str:
        text = text.strip()
        if not text or not settings.enable_google_translate:
            return text
        if not settings.google_translate_api_key:
            raise RuntimeError(
                "Thiếu GOOGLE_TRANSLATE_API_KEY trong .env. "
                "Hãy bật Cloud Translation API hoặc đặt "
                "ENABLE_GOOGLE_TRANSLATE=false để dùng trực tiếp SigLIP2 multilingual."
            )

        endpoint = "https://translation.googleapis.com/language/translate/v2?" + urlencode(
            {"key": settings.google_translate_api_key}
        )
        body = json.dumps(
            {"q": text, "target": "en", "format": "text"},
            ensure_ascii=False,
        ).encode("utf-8")
        request = Request(
            endpoint,
            data=body,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=settings.google_translate_timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            translated = payload["data"]["translations"][0]["translatedText"]
            return html.unescape(str(translated)).strip() or text
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(
                f"Google Translate API lỗi HTTP {exc.code}: {detail}"
            ) from exc
        except (URLError, TimeoutError, KeyError, IndexError, ValueError) as exc:
            raise RuntimeError(f"Không gọi được Google Translate API: {exc}") from exc
