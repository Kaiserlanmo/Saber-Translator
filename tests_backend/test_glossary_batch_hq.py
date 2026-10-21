"""Tests for glossary batching and HQ glossary suggestions."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend_v2.content.translation_constraints import (
    empty_translation_constraints,
    validate_translation_constraints,
)
from src.backend_v2.translation.pipeline import CoreTranslationAlgorithms


def _old_constraints() -> dict:
    return {
        "glossary": {
            "enabled": True,
            "autoExtractEnabled": True,
            "autoExtractPrompt": "old prompt {ocr_text}",
            "entries": [
                {
                    "source": "Alice",
                    "target": "爱丽丝",
                    "note": "",
                    "matchMode": "text",
                }
            ],
        },
        "nonTranslate": {"enabled": False, "entries": []},
    }


def test_old_glossary_document_gets_new_defaults() -> None:
    normalized = validate_translation_constraints(_old_constraints())
    glossary = normalized["glossary"]
    assert glossary["autoExtractBatchPages"] == 5
    assert glossary["hqAutoFillEnabled"] is False
    assert glossary["hqAutoFillPrompt"].strip()
    assert glossary["autoExtractEnabled"] is True


def test_hq_auto_fill_and_auto_extract_are_mutually_exclusive() -> None:
    payload = _old_constraints()
    payload["glossary"]["hqAutoFillEnabled"] = True
    normalized = validate_translation_constraints(payload)
    assert normalized["glossary"]["hqAutoFillEnabled"] is True
    assert normalized["glossary"]["autoExtractEnabled"] is False


def test_auto_extract_batch_pages_is_bounded() -> None:
    payload = _old_constraints()
    payload["glossary"]["autoExtractBatchPages"] = 0
    with pytest.raises(ValueError):
        validate_translation_constraints(payload)


def _hq_config(**overrides):
    config = {
        "target_language": "zh",
        "prompt_content": "主翻译提示词",
        "provider": "siliconflow",
        "model_name": "test-model",
        "custom_base_url": "",
        "compress_vision_images": True,
        "openai_options": {
            "request": {
                "force_json_output": False,
                "temperature": None,
                "extra_body": {},
            },
            "execution": {
                "use_stream": False,
                "rpm_limit": 0,
                "transport_retries": 0,
                "business_retries": 0,
            },
        },
    }
    config.update(overrides)
    return config


def _pages() -> list[dict]:
    return [
        {
            "pageId": "page-1",
            "bubbles": [
                {
                    "bubbleId": "bubble-1",
                    "originalText": "Alice",
                    "translatedText": "",
                    "textDirection": "horizontal",
                }
            ],
        }
    ]


def test_hq_batch_returns_glossary_suggestions() -> None:
    raw = (
        '{"pages":[{"pageId":"p1","bubbles":'
        '[{"bubbleId":"b1","translatedText":"译文"}]}],'
        '"glossary":[{"source":"Alice","target":"爱丽丝","note":"人名"}]}'
    )
    captured = {}

    class FakeExecutor:
        def execute(self, request, *, capability, parser):
            captured["system"] = request.messages[0]["content"]
            return SimpleNamespace(raw_content=raw, parsed=parser(raw))

    pipeline = CoreTranslationAlgorithms.__new__(CoreTranslationAlgorithms)
    with patch(
        "src.shared.openai_execution.OpenAICompatibleSyncExecutor",
        FakeExecutor,
    ), Image.new("RGB", (200, 200), "white") as image:
        result = pipeline.translate_batch(
            _pages(),
            [image],
            _hq_config(
                hq_glossary_enabled=True,
                hq_glossary_prompt="请优先音译人名",
                hq_glossary_entries=[],
            ),
            mode="hq",
        )

    assert "glossary" in captured["system"]
    assert result["pages"]["page-1"]["bubble-1"] == "译文"
    assert result["glossary"][0]["source"] == "Alice"


def test_malformed_glossary_does_not_break_translation() -> None:
    raw = (
        '{"pages":[{"pageId":"p1","bubbles":'
        '[{"bubbleId":"b1","translatedText":"译文"}]}],'
        '"glossary":"not-a-list"}'
    )

    class FakeExecutor:
        def execute(self, request, *, capability, parser):
            return SimpleNamespace(raw_content=raw, parsed=parser(raw))

    pipeline = CoreTranslationAlgorithms.__new__(CoreTranslationAlgorithms)
    with patch(
        "src.shared.openai_execution.OpenAICompatibleSyncExecutor",
        FakeExecutor,
    ), Image.new("RGB", (200, 200), "white") as image:
        result = pipeline.translate_batch(
            _pages(),
            [image],
            _hq_config(
                hq_glossary_enabled=True,
                hq_glossary_prompt="提示词",
                hq_glossary_entries=[],
            ),
            mode="hq",
        )

    assert result["pages"]["page-1"]["bubble-1"] == "译文"
    assert result["glossary"] == []
