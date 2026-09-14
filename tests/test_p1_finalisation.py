"""Focused tests for the metadata-only P1 release finaliser."""

from __future__ import annotations

import collections.abc as c
from pathlib import Path
from types import SimpleNamespace

import pytest

import p1_dataset.finalisation as finalisation
from p1_dataset.finalisation import FinalisationError, finalise_public_corpus
from p1_dataset.publish import UploadOperation
from p1_dataset.validation import PinnedHubClipRetriever

_REVISION = "a" * 40
_DIGEST = "b" * 64
_SHARD = finalisation._Shard("data/train/part-00000.parquet", 10, 2, "c" * 64)


def test_card_update_uses_cas_and_preserves_licence_and_inventory() -> None:
    """Card mode commits only README bytes against the pinned parent."""
    old_head = _REVISION
    new_head = "d" * 40
    readme = b"## Dataset\n\n## Source\n\n## Access\n\n## Licence\n"
    licence = b"unchanged licence"

    class CardHub(_Hub):
        def __init__(self) -> None:
            super().__init__(("README.md", "LICENSE", _SHARD.path))
            self.head = old_head
            self.files: dict[str, bytes] = {
                "README.md": readme,
                "LICENSE": licence,
                _SHARD.path: b"parquet",
            }
            self.operations: tuple[str, ...] = ()

        def create_commit(
            self,
            repo_id: str,
            operations: c.Iterable[UploadOperation],
            **kwargs: object,
        ) -> object:
            del repo_id
            assert kwargs["parent_commit"] == old_head
            self.operations = tuple(item.path_in_repo for item in operations)
            self.files["README.md"] = next(
                item.path.read_bytes() for item in operations
            )
            self.head = new_head
            return SimpleNamespace(commit_id=new_head)

        def get_paths_info(
            self, repo_id: str, paths: list[str], *, repo_type: str, revision: str
        ) -> tuple[object, ...]:
            del repo_id, repo_type, revision
            return tuple({"path": path, "sha256": "e" * 64} for path in paths)

        def list_repo_files(
            self, repo_id: str, *, repo_type: str, revision: str
        ) -> tuple[str, ...]:
            del repo_id, repo_type, revision
            return tuple(self.files)

        def repo_info(self, repo_id: str, *, repo_type: str, revision: str) -> object:
            del repo_id, repo_type, revision
            return SimpleNamespace(private=False, sha=self.head)

        def stream_file(
            self, repo_id: str, path: str, *, repo_type: str, revision: str
        ) -> tuple[bytes, ...]:
            del repo_id, repo_type, revision
            return (self.files[path],)

    hub = CardHub()
    head = finalisation._cas_card_update(
        hub=hub,
        repository="syvai/p1-segments",
        revision=old_head,
        report={"counts": {"rows": 2, "shards": 1}, "audio": {"hours": 1.0}},
        remote_files=("README.md", "LICENSE", _SHARD.path),
    )
    assert head == new_head
    assert hub.operations == ("README.md",)
    assert hub.files["LICENSE"] == licence


def test_finalisation_dry_run_is_metadata_only_and_writes_sanitised_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A successful dry run uses no dataset loader or Hub mutation."""
    hub = _Hub(("README.md", "LICENSE", _SHARD.path))
    monkeypatch.setattr(finalisation, "_read_run_root", lambda *args: _summary())
    monkeypatch.setattr(finalisation, "_check_remote_metadata", lambda *args: None)
    monkeypatch.setattr(finalisation, "_check_manifests", lambda *args: None)
    monkeypatch.setattr(finalisation, "_scan_parquet", lambda *args: _stats())

    report_path = tmp_path / "report.json"
    report = finalise_public_corpus(
        hub=hub,
        run_root=tmp_path,
        revision=_REVISION,
        expected_pipeline_config_sha256=_DIGEST,
        report_path=report_path,
    )

    assert report["pass"] is True
    assert report_path.is_file()
    saved = report_path.read_text(encoding="utf-8")
    assert _SHARD.path not in saved
    assert "e" * 64 not in saved
    assert hub.commits == 0


class _Hub:
    """Small Hub fake used to prove metadata-only orchestration."""

    def __init__(self, files: tuple[str, ...]) -> None:
        self.files = files
        self.commits = 0

    def list_repo_files(
        self, repo_id: str, *, repo_type: str, revision: str
    ) -> tuple[str, ...]:
        del repo_id, repo_type, revision
        return self.files

    def repo_info(self, repo_id: str, *, repo_type: str, revision: str) -> object:
        del repo_id, repo_type
        return SimpleNamespace(private=False, sha=revision)


def _stats() -> dict[str, object]:
    duration = finalisation._Stats()
    duration.add(1_000)
    words = finalisation._Stats()
    words.add(2)
    return {
        "rows": 2,
        "programmes": 1,
        "duration_ms": 2_000,
        "duration": duration,
        "words": words,
        "speakers": {"1": 2},
    }


def _summary() -> dict[str, object]:
    return {
        "shards": (_SHARD,),
        "rows": 2,
        "programme_states": {"purged": 1},
        "batches": {"purged": 1},
    }


def test_finalisation_fails_closed_on_remote_inventory_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A path not represented by the ledgers prevents the report."""
    hub = _Hub(("README.md", "LICENSE", "data/train/other.parquet"))
    monkeypatch.setattr(finalisation, "_read_run_root", lambda *args: _summary())
    with pytest.raises(FinalisationError, match="inventory"):
        finalise_public_corpus(
            hub=hub,
            run_root=tmp_path,
            revision=_REVISION,
            expected_pipeline_config_sha256=_DIGEST,
        )


def test_pinned_retriever_can_require_public_visibility() -> None:
    """The opt-in public policy accepts a public repository."""
    hub = _Hub(("README.md",))
    retriever = PinnedHubClipRetriever(
        hub,
        repository="syvai/p1-segments",
        revision=_REVISION,
        expected_visibility="public",
    )
    retriever.verify_repository()
