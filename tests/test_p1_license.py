"""Tests for the custom layered P1 dataset licence."""

from __future__ import annotations

import hashlib
from pathlib import Path

from hviske.p1_contracts import P1_RUNTIME_CONTRACT

_SOURCE_README_URL = (
    "https://huggingface.co/datasets/syvai/p1/resolve/"
    "449b9c2294026df6d0d37538f279fdec03f565ff/README.md"
)


def test_license_provenance_uses_pinned_source_rights_coordinates() -> None:
    """The pinned Hub source is provenance, not presented as a licence text."""
    contract = P1_RUNTIME_CONTRACT
    assert contract.dataset_license_template_repository == "syvai/p1"
    assert contract.dataset_license_template_revision == (
        "449b9c2294026df6d0d37538f279fdec03f565ff"
    )
    assert contract.dataset_license_template_url == _SOURCE_README_URL
    assert contract.dataset_license_template_sha256 == (
        "133a669aa02adbb3d2ca2ebebae14a811be1703eae7157a75f94b465dcdd1ece"
    )
    assert contract.dataset_license_template_bytes == 2002
    assert "not a textual licence template" in contract.dataset_license_adaptation


def test_target_license_is_the_pinned_custom_layered_license() -> None:
    """The tracked licence bytes and contract digest cannot drift."""
    root = Path(__file__).parents[1]
    target_bytes = (root / "LICENSE-DATASET").read_bytes()
    published_bytes = (root / "LICENSE").read_bytes()

    assert target_bytes == published_bytes
    assert (
        hashlib.sha256(target_bytes).hexdigest()
        == "02f6d0056a6f19f59b57c6a20c49bf7edd90530fa2664bb7e4a286522c981c6e"
        == P1_RUNTIME_CONTRACT.dataset_license_target_sha256
    )
    target_text = target_bytes.decode("utf-8")
    assert "syv.ai ApS" in target_text
    assert "CC BY 4.0" in target_text
    assert "https://creativecommons.org/licenses/by/4.0/legalcode" in target_text
    assert "embedded DR audio" in target_text
    assert "verbatim/source-derived" in target_text
    assert "transcript text" in target_text
    assert "public\nredistribution or sublicensing" in target_text
    assert "ElevenLabs Scribe v2" in target_text
    assert "https://www.dr.dk/om-dr/vilkaar-paa-drdk" in target_text
    assert "https://elevenlabs.io/terms-of-use-eu" in target_text
    assert "https://elevenlabs.io/speech-to-text-terms" in target_text
    assert "CoRal" not in target_text
    assert "Alexandra" not in target_text
    assert "Roest" not in target_text
