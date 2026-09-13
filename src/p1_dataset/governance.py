"""One-shot, metadata-only migration of P1 publication governance."""

from __future__ import annotations

import collections.abc as c
import hashlib
import os
import re
import tempfile
import typing as t
from dataclasses import dataclass
from pathlib import Path

from .hub_diagnostics import classify_hub_error
from .pipeline import PipelineSettings, _publication_lock, build_active_dataset_card
from .publish import (
    HubClient,
    PublicationError,
    UploadOperation,
    _commit_sha,
    _target_head,
    assert_active_governance,
)

_HISTORICAL_LICENSE_SHA256 = (
    "02f6d0056a6f19f59b57c6a20c49bf7edd90530fa2664bb7e4a286522c981c6e"
)
_HISTORICAL_LICENSE_BYTES = 2697


class _GovernanceHubClient(t.Protocol):
    """Metadata and settings surface required by the one-shot migration."""

    def create_commit(
        self,
        repo_id: str,
        operations: c.Iterable[UploadOperation],
        *,
        repo_type: str,
        commit_message: str,
        parent_commit: str | None = None,
    ) -> object: ...

    def list_repo_commits(
        self, repo_id: str, *, repo_type: str, revision: str | None = None
    ) -> c.Iterable[object]: ...

    def list_repo_files(
        self, repo_id: str, *, repo_type: str, revision: str | None = None
    ) -> c.Iterable[str]: ...

    def repo_info(
        self, repo_id: str, *, repo_type: str, revision: str | None = None
    ) -> object: ...

    def stream_file(
        self, repo_id: str, path: str, *, repo_type: str, revision: str
    ) -> c.Iterable[bytes]: ...

    def update_repo_visibility(
        self, repo_id: str, *, repo_type: str, private: bool
    ) -> object: ...


@dataclass(frozen=True)
class GovernanceMigrationReport:
    """Aggregate, non-sensitive evidence from a governance migration."""

    applied: bool
    file_count: int
    readme_sha256: str
    license_sha256: str
    commit_sha256: str | None = None


def migrate_public_governance(
    *,
    api: _GovernanceHubClient,
    settings: PipelineSettings,
    expected_old_head: str,
    apply: bool,
) -> GovernanceMigrationReport:
    """Migrate an existing private P1 target to exact public governance.

    The shared publication lock covers the complete compare-and-swap operation.
    Corpus objects are inventoried by repository path only and are never downloaded.
    Failures before submission, and Hub failures proven to occur before commit
    creation, restore private visibility. An ambiguous submission is left public for
    investigation rather than risking a rollback of a successful commit.

    Args:
        api:
            Injectable Hub client.
        settings:
            Validated active P1 settings.
        expected_old_head:
            Full private target HEAD used as the compare-and-swap parent.
        apply:
            Whether to mutate the target; false performs all old-state checks only.

    Returns:
        Aggregate migration evidence without repository or local paths.

    Raises:
        ValueError:
            If the expected HEAD or configured visibility is invalid.
    """
    if len(expected_old_head) != 40 or any(
        character not in "0123456789abcdef" for character in expected_old_head
    ):
        raise ValueError("expected old HEAD must be a complete lowercase commit SHA")
    if settings.expected_target_visibility != "public":
        raise ValueError("migration target governance must require public visibility")
    with _publication_lock(settings.publish_lock_path):
        return _migrate_public_governance_unlocked(
            api=api, settings=settings, expected_old_head=expected_old_head, apply=apply
        )


def _migrate_public_governance_unlocked(
    *,
    api: _GovernanceHubClient,
    settings: PipelineSettings,
    expected_old_head: str,
    apply: bool,
) -> GovernanceMigrationReport:
    info = api.repo_info(repo_id=settings.target_private_repo, repo_type="dataset")
    old_head = _target_head(
        info, settings.target_private_repo, expected_visibility="private"
    )
    if old_head != expected_old_head:
        raise PublicationError("private target HEAD does not match expected old HEAD")
    old_paths = _inventory_paths(api=api, settings=settings, revision=old_head)
    expected_old_card = _historical_private_card(settings).encode("utf-8")
    old_readme = _read_metadata(
        api=api, settings=settings, path="README.md", revision=old_head
    )
    old_license = _read_metadata(
        api=api, settings=settings, path="LICENSE", revision=old_head
    )
    if old_readme != expected_old_card:
        raise PublicationError("old README does not match historical governance")
    if (
        len(old_license) != _HISTORICAL_LICENSE_BYTES
        or hashlib.sha256(old_license).hexdigest() != _HISTORICAL_LICENSE_SHA256
    ):
        raise PublicationError("old LICENSE does not match historical governance")

    active_card = build_active_dataset_card(settings)
    active_license = settings.active_license_path.read_bytes()
    report = GovernanceMigrationReport(
        applied=False,
        file_count=len(old_paths),
        readme_sha256=hashlib.sha256(active_card.encode("utf-8")).hexdigest(),
        license_sha256=settings.active_governance.license_sha256,
    )
    if not apply:
        return report

    commit_result_available = False
    try:
        with tempfile.TemporaryDirectory(prefix="hviske-p1-governance-") as directory:
            root = Path(directory)
            readme_path = root / "README.md"
            license_path = root / "LICENSE"
            _stage_metadata(
                readme_path=readme_path,
                license_path=license_path,
                active_card=active_card,
                active_license=active_license,
                expected_license_sha256=settings.active_governance.license_sha256,
                expected_license_bytes=settings.active_governance.license_bytes,
            )
            operations = (
                UploadOperation(path_in_repo="README.md", path=readme_path),
                UploadOperation(path_in_repo="LICENSE", path=license_path),
            )
            api.update_repo_visibility(
                settings.target_private_repo, repo_type="dataset", private=False
            )
            public_info = api.repo_info(
                repo_id=settings.target_private_repo, repo_type="dataset"
            )
            public_head = _target_head(
                public_info, settings.target_private_repo, expected_visibility="public"
            )
            if public_head != old_head:
                raise PublicationError("target HEAD changed while switching visibility")
            if (
                _inventory_paths(api=api, settings=settings, revision=public_head)
                != old_paths
            ):
                raise PublicationError(
                    "target inventory changed while switching visibility"
                )
            # A returned result proves that a commit may exist. Until then, a
            # deterministic pre-upload failure can safely be followed by rollback.
            try:
                result = api.create_commit(
                    settings.target_private_repo,
                    operations,
                    repo_type="dataset",
                    commit_message="Migrate P1 dataset to public governance",
                    parent_commit=expected_old_head,
                )
            except Exception as error:
                if _commit_result_may_exist(error):
                    commit_result_available = True
                raise
            commit_result_available = True
    except Exception:
        if not commit_result_available:
            _restore_private(
                api=api,
                settings=settings,
                expected_old_head=expected_old_head,
                expected_paths=old_paths,
            )
        raise

    commit_id = _commit_sha(result)
    _verify_commit_descends(
        api=api,
        settings=settings,
        expected_old_head=expected_old_head,
        new_head=commit_id,
    )
    assert_active_governance(
        t.cast(HubClient, api),
        settings.target_private_repo,
        expected_visibility="public",
        expected_card=active_card,
        expected_license_sha256=settings.active_governance.license_sha256,
        expected_license_bytes=settings.active_governance.license_bytes,
        revision=commit_id,
    )
    new_paths = _inventory_paths(api=api, settings=settings, revision=commit_id)
    if new_paths != old_paths:
        raise PublicationError("governance commit changed repository inventory")
    return GovernanceMigrationReport(
        applied=True,
        file_count=len(new_paths),
        readme_sha256=report.readme_sha256,
        license_sha256=report.license_sha256,
        commit_sha256=commit_id,
    )


def _commit_result_may_exist(error: BaseException) -> bool:
    """Return whether a failed Hub request has an ambiguous commit outcome."""
    diagnostic = classify_hub_error(error)
    if diagnostic.reason in {"missing_uploaded_object", "xet_unavailable"}:
        return False
    return diagnostic.phase not in {"preupload", "lfs_batch", "xet"}


def _restore_private(
    *,
    api: _GovernanceHubClient,
    settings: PipelineSettings,
    expected_old_head: str,
    expected_paths: tuple[str, ...],
) -> None:
    """Restore private visibility and prove that the old tree is still intact.

    Raises:
        PublicationError:
            If visibility, HEAD, or the repository inventory cannot be restored.
    """
    api.update_repo_visibility(
        settings.target_private_repo, repo_type="dataset", private=True
    )
    info = api.repo_info(repo_id=settings.target_private_repo, repo_type="dataset")
    restored_head = _target_head(
        info, settings.target_private_repo, expected_visibility="private"
    )
    if restored_head != expected_old_head:
        raise PublicationError("private visibility rollback did not restore HEAD")
    if (
        _inventory_paths(api=api, settings=settings, revision=expected_old_head)
        != expected_paths
    ):
        raise PublicationError(
            "private visibility rollback changed repository inventory"
        )
    if _read_metadata(
        api=api, settings=settings, path="README.md", revision=expected_old_head
    ) != _historical_private_card(settings).encode("utf-8"):
        raise PublicationError("private visibility rollback changed README")
    restored_license = _read_metadata(
        api=api, settings=settings, path="LICENSE", revision=expected_old_head
    )
    if (
        len(restored_license) != _HISTORICAL_LICENSE_BYTES
        or hashlib.sha256(restored_license).hexdigest() != _HISTORICAL_LICENSE_SHA256
    ):
        raise PublicationError("private visibility rollback changed LICENSE")


def _stage_metadata(
    *,
    readme_path: Path,
    license_path: Path,
    active_card: str,
    active_license: bytes,
    expected_license_sha256: str,
    expected_license_bytes: int,
) -> None:
    """Durably stage and re-read both metadata files before a visibility change.

    Raises:
        PublicationError:
            If either staged file fails validation.
    """
    with readme_path.open("w", encoding="utf-8") as handle:
        handle.write(active_card)
        handle.flush()
        os.fsync(handle.fileno())
    with license_path.open("wb") as handle:
        handle.write(active_license)
        handle.flush()
        os.fsync(handle.fileno())
    directory_fd = os.open(readme_path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    staged_license = license_path.read_bytes()
    if (
        not readme_path.is_file()
        or readme_path.is_symlink()
        or not license_path.is_file()
        or license_path.is_symlink()
        or readme_path.read_text(encoding="utf-8") != active_card
        or staged_license != active_license
        or len(staged_license) != expected_license_bytes
        or hashlib.sha256(staged_license).hexdigest() != expected_license_sha256
    ):
        raise PublicationError("local governance staging validation failed")


def _verify_commit_descends(
    *,
    api: _GovernanceHubClient,
    settings: PipelineSettings,
    expected_old_head: str,
    new_head: str,
) -> None:
    """Prove the metadata commit is the direct child of the expected old HEAD.

    Raises:
        PublicationError:
            If current HEAD or Hub ancestry does not prove the expected parent.
    """
    current_info = api.repo_info(
        repo_id=settings.target_private_repo, repo_type="dataset"
    )
    current_head = _target_head(
        current_info, settings.target_private_repo, expected_visibility="public"
    )
    if current_head != new_head:
        raise PublicationError("Hub current HEAD differs from returned commit")
    commits = tuple(
        api.list_repo_commits(
            settings.target_private_repo, repo_type="dataset", revision=new_head
        )
    )
    commit_ids = tuple(_commit_sha(item) for item in commits)
    if len(commit_ids) < 2 or commit_ids[0] != new_head:
        raise PublicationError("Hub returned unknown commit ancestry")
    parent_value = _value(commits[0], "parents", "parent_commit", "parent")
    if parent_value is not None:
        parents = _parent_shas(parent_value)
        if parents != (expected_old_head,):
            raise PublicationError("metadata commit has the wrong parent")
    elif commit_ids[1] != expected_old_head:
        raise PublicationError("metadata commit ancestry is not directly verifiable")
    elif len(set(commit_ids[:2])) != 2:
        raise PublicationError("Hub returned ambiguous commit ancestry")


def _parent_shas(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        values: tuple[object, ...] = (value,)
    elif isinstance(value, c.Iterable) and not isinstance(value, (bytes, str)):
        values = tuple(value)
    else:
        values = (value,)
    result: list[str] = []
    for item in values:
        candidate = _value(item, "commit_id", "oid", "sha")
        if candidate is None and type(item) is str:
            candidate = item
        if not isinstance(candidate, str) or not _COMMIT_SHA.fullmatch(candidate):
            return ()
        result.append(candidate)
    return tuple(result)


def _value(value: object, *names: str) -> object:
    for name in names:
        if isinstance(value, dict) and name in value:
            return value[name]
        candidate = getattr(value, name, None)
        if candidate is not None:
            return candidate
    return None


_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def _historical_private_card(settings: PipelineSettings) -> str:
    card = build_active_dataset_card(settings)
    card = card.replace(
        "This dataset is publicly accessible. Load it with an immutable dataset "
        "commit revision (never `main`):",
        "This is a private dataset for authorised users. Load it with an immutable "
        "dataset commit revision (not `main`):",
    )
    card = card.replace(
        "```\n\n## Licence",
        "```\n\nAccess is restricted to authorised syv.ai members.\n\n## Licence",
    )
    card = card.replace(
        "Public availability does not place excluded content under an open licence "
        "or independently grant downstream rights. Users need relevant permissions "
        "or another basis under applicable law.",
        "Private access does not grant public redistribution or sublicensing of the "
        "excluded content.",
    )
    return card


def _inventory_paths(
    *, api: _GovernanceHubClient, settings: PipelineSettings, revision: str
) -> tuple[str, ...]:
    paths = tuple(
        sorted(
            str(path)
            for path in api.list_repo_files(
                repo_id=settings.target_private_repo,
                repo_type="dataset",
                revision=revision,
            )
        )
    )
    if "README.md" not in paths or "LICENSE" not in paths:
        raise PublicationError("target metadata inventory is incomplete")
    return paths


def _read_metadata(
    *, api: _GovernanceHubClient, settings: PipelineSettings, path: str, revision: str
) -> bytes:
    try:
        return b"".join(
            api.stream_file(
                settings.target_private_repo,
                path,
                repo_type="dataset",
                revision=revision,
            )
        )
    except Exception as error:
        raise PublicationError("cannot read target governance metadata") from error
