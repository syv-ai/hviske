"""Tests for the immutable P1 dataset licence adaptation."""

from __future__ import annotations

import hashlib
from pathlib import Path

from hviske.p1_contracts import P1_RUNTIME_CONTRACT

_PINNED_TEMPLATE_URL = (
    "https://huggingface.co/datasets/CoRal-project/coral-v3/resolve/"
    "01f7c93c21fc9dec87fe9f7149c79569cc433f08/LICENSE"
)
_SOURCE_SEQUENCE_TEXT = (
    "The Licensed Material (as defined below) is made available to You by Alexandra\n"
    "Instituttet A/S, Åbogade 34, 8200 Aarhus N, Denmark"
)
_TARGET_SEQUENCE_TEXT = (
    "The Licensed Material (as defined below) is made available to You by syv.ai ApS,\n"
    "Rosenvængets Allé 11, 1. tv, 2100 København Ø, Denmark"
)
_SOURCE_SEQUENCE_BYTES = _SOURCE_SEQUENCE_TEXT.encode("utf-8")
_TARGET_SEQUENCE_BYTES = _TARGET_SEQUENCE_TEXT.encode("utf-8")


def test_target_license_is_exactly_the_pinned_source_adaptation() -> None:
    """Only the exact licensor identity bytes differ from the pinned source."""
    root = Path(__file__).parents[1]
    source_bytes = (root / "tests/fixtures/p1/LICENSE.source").read_bytes()
    target_bytes = (root / "LICENSE-DATASET").read_bytes()

    assert len(source_bytes) == P1_RUNTIME_CONTRACT.dataset_license_template_bytes
    assert (
        hashlib.sha256(source_bytes).hexdigest()
        == "ee93c98df9a894464d1c042b66c6039543776c463d06d1a6e93f176e08ca67bd"
        == P1_RUNTIME_CONTRACT.dataset_license_template_sha256
    )
    assert source_bytes.count(_SOURCE_SEQUENCE_BYTES) == 1
    expected_target_bytes = source_bytes.replace(
        _SOURCE_SEQUENCE_BYTES, _TARGET_SEQUENCE_BYTES
    )
    assert target_bytes == expected_target_bytes
    assert (
        hashlib.sha256(target_bytes).hexdigest()
        == "e06010caf8ea36292a241339c08eedc7ea399778954cf9a546e989d421a48cd6"
        == P1_RUNTIME_CONTRACT.dataset_license_target_sha256
    )

    source_text = source_bytes.decode("utf-8")
    target_text = target_bytes.decode("utf-8")
    assert source_text.count(_SOURCE_SEQUENCE_TEXT) == 1
    assert target_text.count(_TARGET_SEQUENCE_TEXT) == 1
    assert _SOURCE_SEQUENCE_TEXT in P1_RUNTIME_CONTRACT.dataset_license_adaptation
    assert _TARGET_SEQUENCE_TEXT in P1_RUNTIME_CONTRACT.dataset_license_adaptation
    assert "The place of arbitration shall be Aarhus, Denmark." in target_text
    assert P1_RUNTIME_CONTRACT.dataset_license_template_url == _PINNED_TEMPLATE_URL
