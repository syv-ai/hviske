"""Tests for the one-shot P1 public-governance migration."""

from __future__ import annotations

import collections.abc as c
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from omegaconf import DictConfig, OmegaConf

from p1_dataset.governance import _historical_private_card, migrate_public_governance
from p1_dataset.pipeline import PipelineSettings, build_active_dataset_card
from p1_dataset.publish import PublicationError, UploadOperation

_OLD_HEAD = "a" * 40
_NEW_HEAD = "b" * 40


def test_ambiguous_commit_is_never_rolled_back(tmp_path: Path) -> None:
    """An ambiguous returned commit failure leaves public state for investigation."""
    active = settings(tmp_path)
    hub = MigrationHub(active)
    hub.ambiguous_commit = True
    with pytest.raises(RuntimeError, match="ambiguous transport failure"):
        migrate_public_governance(
            api=hub, settings=active, expected_old_head=_OLD_HEAD, apply=True
        )
    assert hub.private is False
    assert hub.events[-1] == "commit"
    assert "private" not in hub.events


class MigrationHub:
    """Stateful metadata-only Hub fake for migration ordering."""

    def __init__(self, settings: PipelineSettings) -> None:
        """Create a private target containing historical metadata and one data path."""
        historical = Path(
            "src/p1_dataset/historical/p1-private-governance-license.txt"
        ).read_bytes()
        self.private: object = True
        self.sha = _OLD_HEAD
        self.files = {
            "README.md": _historical_private_card(settings).encode("utf-8"),
            "LICENSE": historical,
            "data/train/part.parquet": b"never read",
        }
        self.events: list[str] = []
        self.operations: tuple[str, ...] = ()
        self.fail_inventory_after_switch = False
        self.ambiguous_commit = False

    def create_commit(
        self,
        repo_id: str,
        operations: c.Iterable[UploadOperation],
        *,
        repo_type: str,
        commit_message: str,
        parent_commit: str | None = None,
    ) -> object:
        """Apply exactly the supplied additive metadata operations.

        Returns:
            Fake immutable commit metadata.

        Raises:
            RuntimeError:
                When configured to simulate an ambiguous commit response.
        """
        del repo_id, repo_type, commit_message
        assert parent_commit == _OLD_HEAD
        self.events.append("commit")
        operations = tuple(operations)
        self.operations = tuple(item.path_in_repo for item in operations)
        if self.ambiguous_commit:
            raise RuntimeError("ambiguous transport failure")
        for operation in operations:
            self.files[operation.path_in_repo] = operation.path.read_bytes()
        self.sha = _NEW_HEAD
        return SimpleNamespace(commit_id=_NEW_HEAD)

    def list_repo_files(
        self, repo_id: str, *, repo_type: str, revision: str | None = None
    ) -> c.Iterable[str]:
        """Return path metadata without reading file payloads.

        Raises:
            RuntimeError:
                When configured to simulate a post-switch metadata failure.
        """
        del repo_id, repo_type, revision
        self.events.append("inventory")
        if self.fail_inventory_after_switch and self.private is False:
            raise RuntimeError("metadata failure")
        return tuple(self.files)

    def repo_info(
        self, repo_id: str, *, repo_type: str, revision: str | None = None
    ) -> object:
        """Return current visibility and HEAD."""
        del repo_id, repo_type, revision
        self.events.append("info")
        return SimpleNamespace(private=self.private, sha=self.sha)

    def stream_file(
        self, repo_id: str, path: str, *, repo_type: str, revision: str
    ) -> c.Iterable[bytes]:
        """Return governance metadata only."""
        del repo_id, repo_type, revision
        self.events.append(f"read:{path}")
        assert not path.startswith("data/")
        return (self.files[path],)

    def update_repo_visibility(
        self, repo_id: str, *, repo_type: str, private: bool
    ) -> object:
        """Record and apply one visibility operation.

        Returns:
            Fake visibility metadata.
        """
        del repo_id, repo_type
        self.events.append("private" if private else "public")
        self.private = private
        return SimpleNamespace(private=private)


def settings(tmp_path: Path) -> PipelineSettings:
    """Load active settings with a test-owned publication lock.

    Returns:
        Validated P1 settings.
    """
    config = OmegaConf.load("config/p1_segments.yaml")
    config.runtime.scratch_root = str(tmp_path)
    return PipelineSettings.from_config(cast(DictConfig, config))


def test_migration_cas_and_precommit_failure_restore_private(tmp_path: Path) -> None:
    """CAS fails closed and a post-switch metadata failure restores private."""
    active = settings(tmp_path)
    hub = MigrationHub(active)
    with pytest.raises(PublicationError, match="HEAD"):
        migrate_public_governance(
            api=hub, settings=active, expected_old_head="c" * 40, apply=True
        )
    assert hub.private is True
    assert "public" not in hub.events

    hub = MigrationHub(active)
    hub.fail_inventory_after_switch = True
    with pytest.raises(RuntimeError, match="metadata failure"):
        migrate_public_governance(
            api=hub, settings=active, expected_old_head=_OLD_HEAD, apply=True
        )
    assert hub.events[-1] == "private"
    assert hub.private is True
    assert "commit" not in hub.events


def test_migration_dry_run_is_metadata_only_and_preserves_state(tmp_path: Path) -> None:
    """Dry-run validates historical state without switching or committing."""
    active = settings(tmp_path)
    hub = MigrationHub(active)
    report = migrate_public_governance(
        api=hub, settings=active, expected_old_head=_OLD_HEAD, apply=False
    )
    assert not report.applied
    assert report.file_count == 3
    assert hub.private is True
    assert "public" not in hub.events
    assert "commit" not in hub.events
    assert hub.files["data/train/part.parquet"] == b"never read"


def test_migration_orders_switch_then_two_metadata_operations(tmp_path: Path) -> None:
    """Apply uses CAS and preserves the complete repository inventory."""
    active = settings(tmp_path)
    hub = MigrationHub(active)
    before = tuple(hub.files)
    report = migrate_public_governance(
        api=hub, settings=active, expected_old_head=_OLD_HEAD, apply=True
    )
    assert report.applied
    assert hub.private is False
    assert hub.operations == ("README.md", "LICENSE")
    assert hub.events.index("public") < hub.events.index("commit")
    assert tuple(hub.files) == before
    assert hub.files["README.md"] == build_active_dataset_card(active).encode("utf-8")
    assert hub.files["LICENSE"] == active.active_license_path.read_bytes()
    assert hub.files["data/train/part.parquet"] == b"never read"
