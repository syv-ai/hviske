"""One-shot, metadata-only migration of P1 publication governance."""

from __future__ import annotations

import collections.abc as c
import hashlib
import tempfile
import typing as t
from dataclasses import dataclass
from pathlib import Path

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
    A failure before commit submission restores private visibility. Once submission
    starts, any exception is treated as ambiguous and visibility is not rolled back.

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

    try:
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
    except Exception:
        api.update_repo_visibility(
            settings.target_private_repo, repo_type="dataset", private=True
        )
        raise

    with tempfile.TemporaryDirectory(prefix="hviske-p1-governance-") as directory:
        root = Path(directory)
        readme_path = root / "README.md"
        license_path = root / "LICENSE"
        readme_path.write_text(active_card, encoding="utf-8")
        license_path.write_bytes(active_license)
        operations = (
            UploadOperation(path_in_repo="README.md", path=readme_path),
            UploadOperation(path_in_repo="LICENSE", path=license_path),
        )
        # From this point a transport error may hide a successful commit. Never
        # automatically restore private visibility after an ambiguous submission.
        result = api.create_commit(
            settings.target_private_repo,
            operations,
            repo_type="dataset",
            commit_message="Migrate P1 dataset to public governance",
            parent_commit=expected_old_head,
        )
    commit_id = _commit_sha(result)
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
