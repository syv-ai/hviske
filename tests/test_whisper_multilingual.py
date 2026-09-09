"""Regression coverage for multilingual Whisper label prefixes."""

import numpy as np
import pytest
from transformers import WhisperProcessor

from hviske.data import process_example

WHISPER_REVISION = "169d4a4341b33bc18d8881c4b69c2e104e1cc0af"


def test_real_whisper_processor_uses_each_example_language(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Danish and English examples receive their own Whisper language token."""
    monkeypatch.setattr("hviske.data.download_background_noises", lambda: None)
    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-tiny", revision=WHISPER_REVISION
    )

    labels = {}
    for language in ("da", "en"):
        example = process_example(
            example={
                "text": "Hello",
                "language": language,
                "audio": {
                    "array": np.zeros(16_000, dtype=np.float32),
                    "sampling_rate": 16_000,
                },
            },
            characters_to_keep=None,
            conversion_dict={},
            text_column="text",
            audio_column="audio",
            lower_case=False,
            convert_numerals=False,
            processor=processor,
            normalise_audio=False,
            augment_audio=False,
            language="da",
            language_column="language",
        )
        labels[language] = example["labels"]

    da_id = processor.tokenizer.convert_tokens_to_ids("<|da|>")
    en_id = processor.tokenizer.convert_tokens_to_ids("<|en|>")
    assert labels["da"][1] == da_id
    assert labels["en"][1] == en_id
    assert da_id != en_id
