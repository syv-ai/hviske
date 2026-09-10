"""Tests for the immutable P1 dataset licence adaptation."""

from __future__ import annotations

import hashlib
import unicodedata
from pathlib import Path

from hviske.p1_contracts import P1_RUNTIME_CONTRACT

_SOURCE_IDENTITY = "Alexandra\nInstituttet A/S, Åbogade 34, 8200 Aarhus N, Denmark"
_TARGET_IDENTITY = "syv.ai ApS, Rosenvængets Allé 11, 1. tv, 2100 København Ø, Denmark"


def test_target_license_is_exactly_the_pinned_source_adaptation() -> None:
    """Only the licensor identity differs from the pinned source licence."""
    target_path = Path(__file__).parents[1] / "LICENSE-DATASET"
    adapted = _normalise_license(target_path.read_text(encoding="utf-8"))
    assert adapted.count(_TARGET_IDENTITY) == 1

    source = adapted.replace(_TARGET_IDENTITY, _SOURCE_IDENTITY)
    assert source.count(_SOURCE_IDENTITY) == 1
    assert (
        len(source.encode("utf-8"))
        == 14112
        == P1_RUNTIME_CONTRACT.dataset_license_template_bytes
    )
    assert (
        hashlib.sha256(source.encode("utf-8")).hexdigest()
        == "ee93c98df9a894464d1c042b66c6039543776c463d06d1a6e93f176e08ca67bd"
        == P1_RUNTIME_CONTRACT.dataset_license_template_sha256
    )
    assert (
        hashlib.sha256(adapted.encode("utf-8")).hexdigest()
        == "3c657d8d41bfbe131e2e24afd52004b1ddc10e4438edd97725c9df187ec5aa6f"
        == P1_RUNTIME_CONTRACT.dataset_license_target_sha256
    )
    assert adapted == source.replace(_SOURCE_IDENTITY, _TARGET_IDENTITY)
    assert "The place of arbitration shall be Aarhus, Denmark." in adapted


def _normalise_license(content: str) -> str:
    """Normalise transport differences without changing licence content.

    Returns:
        NFC-normalised licence text with Unix line endings.
    """
    return unicodedata.normalize("NFC", content).replace("\r\n", "\n")
