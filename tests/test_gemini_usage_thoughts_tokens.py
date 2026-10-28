"""Gemini reports thinking tokens apart from the answer's, and bills both as output."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

USAGE = {
    "promptTokenCount": 31,
    "candidatesTokenCount": 10,
    "thoughtsTokenCount": 40,
    "totalTokenCount": 81,
}


class _StreamResponse:
    def iter_lines(self):
        chunk = {
            "candidates": [{"content": {"parts": [{"text": "hi"}]}, "finishReason": "STOP"}],
            "usageMetadata": USAGE,
        }
        yield b"data: " + json.dumps(chunk).encode()


class _VisionResponse:
    status_code = 200
    text = ""

    def json(self):
        return {
            "candidates": [{"content": {"parts": [{"text": "a cat"}]}}],
            "usageMetadata": USAGE,
        }


def _bot():
    from models.gemini.google_gemini_bot import GoogleGeminiBot

    return GoogleGeminiBot.__new__(GoogleGeminiBot)


def test_stream_final_chunk_counts_thinking_tokens_as_completion():
    chunks = list(_bot()._handle_gemini_rest_stream_response(_StreamResponse(), "gemini-2.5-flash"))

    assert chunks[-1]["usage"] == {"prompt_tokens": 31, "completion_tokens": 50, "total_tokens": 81}


def test_vision_usage_counts_thinking_tokens_as_completion(monkeypatch):
    from config import conf
    from models.gemini import google_gemini_bot

    monkeypatch.setitem(conf(), "gemini_api_key", "test")
    bot = _bot()
    monkeypatch.setattr(
        google_gemini_bot.requests, "post", lambda *args, **kwargs: _VisionResponse()
    )
    monkeypatch.setattr(
        google_gemini_bot.GoogleGeminiBot,
        "_build_inline_part_from_image_url",
        lambda self, image: {"inlineData": {"mimeType": "image/png", "data": ""}},
    )

    result = bot.call_vision("data:image/png;base64,AAAA", "what is this?", model="gemini-2.5-flash")

    assert result["usage"] == {"prompt_tokens": 31, "completion_tokens": 50, "total_tokens": 81}


def test_missing_thinking_tokens_leave_the_answer_count_unchanged():
    from models.gemini.google_gemini_bot import _completion_tokens

    assert _completion_tokens({"candidatesTokenCount": 10}) == 10
    assert _completion_tokens({}) == 0
