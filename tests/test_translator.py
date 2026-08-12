import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

# Keep this unit test independent from the optional project environment setup.
sys.modules.setdefault("dotenv", SimpleNamespace(load_dotenv=lambda *_args, **_kwargs: None))

from src.translator import Translator


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(
            {"data": {"translations": [{"translatedText": "a motorbike"}]}}
        ).encode("utf-8")


class TranslatorTests(unittest.TestCase):
    def setUp(self):
        Translator.to_english.cache_clear()

    def test_google_translate_response(self):
        fake_settings = SimpleNamespace(
            enable_google_translate=True,
            google_translate_api_key="test-key",
            google_translate_timeout=5.0,
        )
        with patch("src.translator.settings", fake_settings), patch(
            "src.translator.urlopen", return_value=FakeResponse()
        ) as mocked:
            self.assertEqual(Translator().to_english("một xe máy"), "a motorbike")
            self.assertEqual(mocked.call_count, 1)

    def test_disabled_uses_multilingual_query(self):
        fake_settings = SimpleNamespace(
            enable_google_translate=False,
            google_translate_api_key="",
            google_translate_timeout=5.0,
        )
        with patch("src.translator.settings", fake_settings):
            self.assertEqual(Translator().to_english("một người"), "một người")


if __name__ == "__main__":
    unittest.main()
