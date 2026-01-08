import types

import pytest

from backend import app


def test_validate_and_process_subtitles_basic(monkeypatch):
    # Force deterministic language detection
    monkeypatch.setattr(app, "detect", lambda _: "en")

    raw = {"text": "Hello world. This is a test.", "language": "en"}
    result = app.validate_and_process_subtitles(raw)

    assert result is not None
    assert result["text"].startswith("Hello world")
    assert result["detected_language"] == "en"
    assert result["source"] == "youtube_subtitles"


def test_extract_sentences_with_timestamps_from_subtitles(monkeypatch):
    # Use deterministic language detection
    monkeypatch.setattr(app, "detect", lambda _: "en")

    response = {"text": "First sentence. Second sentence."}
    sentences = app.extract_sentences_with_timestamps(response)

    assert len(sentences) == 2
    assert sentences[0]["text"].startswith("First sentence")
    # Durations are synthetic but should be increasing
    assert sentences[0]["start_time"] == 0
    assert sentences[0]["end_time"] > sentences[0]["start_time"]
    assert sentences[1]["start_time"] >= sentences[0]["end_time"]


def test_extract_sentences_with_timestamps_from_segments():
    response = {
        "segments": [
            {"text": "Hello", "start": 0.0, "end": 1.0},
            {"text": "World", "start": 1.0, "end": 2.0},
        ]
    }
    sentences = app.extract_sentences_with_timestamps(response)

    assert len(sentences) == 2
    assert sentences[0]["text"] == "Hello"
    assert sentences[1]["text"] == "World"
    assert sentences[1]["start_time"] == 1.0
    assert sentences[1]["end_time"] == 2.0
