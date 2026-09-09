"""Private, verified publication of P1 dataset shards."""

from __future__ import annotations

import collections.abc as c
import hashlib
import json
import os
import re
import tempfile
import typing as t
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from datasets import load_dataset
from huggingface_hub import CommitOperationAdd, HfApi, HfFileSystem, hf_hub_url
from huggingface_hub.utils import RepositoryNotFoundError

from .p1_contracts import (
    OUTPUT_SCHEMA,
    BatchEvidence,
    LedgerState,
    RejectionCategory,
    ShardEvidence,
)

if t.TYPE_CHECKING:
    from .p1_ledger import Ledger

_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CREDENTIAL_KEYS = re.compile(
    r"(?:token|secret|password|credential|authorization|api[_-]?key)", re.I
)
_CREDENTIAL_VALUES = re.compile(r"(?:hf_[A-Za-z0-9_-]{10,}|sk-[A-Za-z0-9_-]{10,})")


@dataclass(frozen=True)
class LocalShard:
    """A local Parquet shard and its metadata-only publication identity."""

    path: Path
    repo_path: str
    row_count: int


def build_dataset_card(
    *,
    source_provenance: str,
    permitted_use: str,
    private_access_terms: str,
    alignment_method: str,
    field_schema: str,
    known_limitations: str,
    rejection_policy: str,
    source_revisions: str,
    model_revisions: str,
) -> str:
    """Build the required metadata-only private P1 dataset card.

    Args:
        source_provenance:
            Description of the source data and its provenance.
        permitted_use:
            Permitted uses of the derived dataset.
        private_access_terms:
            Terms governing access to the private repository.
        alignment_method:
            Alignment and segmentation method.
        field_schema:
            Published training-row schema.
        known_limitations:
            Known quality or coverage limitations.
        rejection_policy:
            Policy and categories for rejected material.
        source_revisions:
            Immutable source dataset revisions.
        model_revisions:
            Immutable VAD, CTC, and other model revisions.

    Returns:
        Dataset card Markdown with no credential-bearing metadata.
    """
    values = {
        "Source provenance": source_provenance,
        "Permitted use": permitted_use,
        "Private-access terms": private_access_terms,
        "Alignment method": alignment_method,
        "Field schema": field_schema,
        "Known limitations": known_limitations,
        "Rejection policy": rejection_policy,
        "Source revisions": source_revisions,
        "Model revisions": model_revisions,
    }
    _assert_safe_metadata(values)
    sections = [f"## {name}\n\n{value}" for name, value in values.items()]
    sections.append(
        "## Redistribution\n\n"
        "No public redistribution grant is provided. This dataset remains private."
    )
    return "# P1 segmented Danish speech\n\n" + "\n\n".join(sections) + "\n"


def _assert_safe_metadata(value: object, token: str | None = None) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if _CREDENTIAL_KEYS.search(str(key)):
                raise PublicationError(
                    "credentials are forbidden in publication metadata"
                )
            _assert_safe_metadata(child, token)
    elif isinstance(value, (list, tuple, set)):
        for child in value:
            _assert_safe_metadata(child, token)
    elif token and token in str(value):
        raise PublicationError("credentials are forbidden in publication metadata")
    elif isinstance(value, str) and _CREDENTIAL_VALUES.search(value):
        raise PublicationError("credentials are forbidden in publication metadata")


class PublicationError(RuntimeError):
    """Base error for an unsafe or unverified publication."""


@dataclass(frozen=True)
class UploadOperation:
    """One explicitly allow-listed local file and its remote path."""

    path_in_repo: str
    path: Path


class HfApiAdapter:
    """Production adapter around :mod:`huggingface_hub`.

    Remote checksum fallback uses ``HfFileSystem.open`` and therefore never
    creates a retained download in the publisher's scratch directory.
    """

    def __init__(self, token: str | bool | None = None) -> None:
        """Create an authenticated adapter.

        Args:
            token (optional):
                Hugging Face token, or ``True`` to use the configured session.
        """
        self._token = token if token is not None else True
        self._api = HfApi(token=self._token)
        self._filesystem = HfFileSystem(token=self._token)

    def create_commit(
        self,
        repo_id: str,
        operations: c.Iterable[UploadOperation],
        *,
        repo_type: str,
        commit_message: str,
        parent_commit: str | None = None,
    ) -> object:
        """Commit explicit local files to the dataset repository.

        Returns:
            The Hub commit response.
        """
        hub_operations = [
            CommitOperationAdd(
                path_in_repo=operation.path_in_repo, path_or_fileobj=operation.path
            )
            for operation in operations
        ]
        return self._api.create_commit(
            repo_id=repo_id,
            operations=hub_operations,
            repo_type=repo_type,
            commit_message=commit_message,
            parent_commit=parent_commit,
            token=self._token,
        )

    def create_repo(
        self, repo_id: str, *, repo_type: str, private: bool, exist_ok: bool
    ) -> object:
        """Create a private dataset repository.

        Returns:
            The Hub repository response.
        """
        return self._api.create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            private=private,
            exist_ok=exist_ok,
            token=self._token,
        )

    def get_paths_info(
        self, repo_id: str, paths: list[str], *, repo_type: str, revision: str
    ) -> c.Iterable[object]:
        """Return immutable-revision file metadata."""
        return self._api.get_paths_info(
            repo_id=repo_id,
            paths=paths,
            repo_type=repo_type,
            revision=revision,
            token=self._token,
        )

    def list_repo_files(
        self, repo_id: str, *, repo_type: str, revision: str | None = None
    ) -> c.Iterable[str]:
        """List paths at a revision for collision checks.

        Returns:
            Repository-relative paths.
        """
        return self._api.list_repo_files(
            repo_id=repo_id, repo_type=repo_type, revision=revision, token=self._token
        )

    def load_dataset(
        self, repo_id: str, *, shard_path: str, revision: str, streaming: bool
    ) -> object:
        """Open one remote Parquet shard with Datasets streaming enabled.

        Returns:
            The streaming Datasets object.

        Raises:
            ValueError:
                If streaming mode is disabled.
        """
        if not streaming:
            raise ValueError("P1 verification requires streaming=True")
        url = hf_hub_url(repo_id, shard_path, repo_type="dataset", revision=revision)
        return load_dataset(
            "parquet",
            data_files={"train": url},
            split="train",
            streaming=True,
            token=self._token,
        )

    def repo_info(
        self, repo_id: str, *, repo_type: str, revision: str | None = None
    ) -> object:
        """Return Hub repository metadata."""
        return self._api.repo_info(
            repo_id=repo_id, repo_type=repo_type, revision=revision, token=self._token
        )

    def stream_file(
        self, repo_id: str, path: str, *, repo_type: str, revision: str
    ) -> c.Iterable[bytes]:
        """Yield chunks from a Hub file without a local download.

        Returns:
            An iterator of remote byte chunks.

        Raises:
            ValueError:
                If a non-dataset repository is requested.
        """
        if repo_type != "dataset":
            raise ValueError("P1 publication only supports dataset repositories")

        def chunks() -> c.Iterator[bytes]:
            handle = self._filesystem.open(
                f"{repo_id}/{path}", mode="rb", revision=revision
            )
            try:
                while chunk := handle.read(1024 * 1024):
                    yield chunk
            finally:
                handle.close()

        return chunks()


class HubClient(t.Protocol):
    """The small Hub surface needed by the publisher."""

    def create_commit(
        self,
        repo_id: str,
        operations: c.Iterable[UploadOperation],
        *,
        repo_type: str,
        commit_message: str,
        parent_commit: str | None = None,
    ) -> object:
        """Create a commit from explicit files."""
        ...

    def create_repo(
        self, repo_id: str, *, repo_type: str, private: bool, exist_ok: bool
    ) -> object:
        """Create a repository."""
        ...

    def get_paths_info(
        self, repo_id: str, paths: list[str], *, repo_type: str, revision: str
    ) -> c.Iterable[object]:
        """Return metadata for paths at an immutable revision."""
        ...

    def list_repo_files(
        self, repo_id: str, *, repo_type: str, revision: str | None = None
    ) -> c.Iterable[str]:
        """List paths at a revision for collision checks."""
        ...

    def load_dataset(
        self, repo_id: str, *, shard_path: str, revision: str, streaming: bool
    ) -> object:
        """Open one remote Parquet shard as a streaming dataset."""
        ...

    def repo_info(
        self, repo_id: str, *, repo_type: str, revision: str | None = None
    ) -> object:
        """Return repository metadata."""
        ...

    def stream_file(
        self, repo_id: str, path: str, *, repo_type: str, revision: str
    ) -> c.Iterable[bytes]:
        """Stream a remote object without retaining it locally."""
        ...


def initialise_private_dataset(
    api: HubClient,
    repo_id: str,
    *,
    card: str,
    gitattributes: str = "*.parquet filter=lfs diff=lfs merge=lfs -text\n",
    token: str | None = None,
) -> str | None:
    """Create and initialise a private dataset repository.

    Args:
        api:
            Injectable Hub client.
        repo_id:
            Dataset repository identifier.
        card:
            Dataset card Markdown, without credentials.
        gitattributes (optional):
            Initial Git attributes content.
        token (optional):
            Authentication token, used only to reject accidental card leakage.

    Returns:
        The immutable initialisation commit SHA, if a commit was made.
    """
    _assert_safe_metadata(card, token=token)
    _assert_safe_metadata(gitattributes, token=token)
    try:
        info = api.repo_info(repo_id=repo_id, repo_type="dataset")
    except RepositoryNotFoundError:
        api.create_repo(
            repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True
        )
        info = api.repo_info(repo_id=repo_id, repo_type="dataset")
    _assert_private(info, repo_id)

    with tempfile.TemporaryDirectory(prefix="hviske-p1-card-") as directory:
        root = Path(directory)
        card_path = root / "README.md"
        attrs_path = root / ".gitattributes"
        card_path.write_text(card, encoding="utf-8")
        attrs_path.write_text(gitattributes, encoding="utf-8")
        commit = _mutate_commit(
            api,
            repo_id,
            operations=(
                UploadOperation(path_in_repo="README.md", path=card_path),
                UploadOperation(path_in_repo=".gitattributes", path=attrs_path),
            ),
            message="Initialise private P1 dataset",
        )
    return _commit_sha(commit)


def _assert_private(info: object, repo_id: str) -> None:
    if _value(info, "private") is not True:
        raise PrivacyError(
            f"private-only publication refuses unknown/public repository {repo_id!r}"
        )


class PrivacyError(PublicationError):
    """Raised when the destination is not demonstrably private."""


def _value(value: object, *names: str) -> object:
    for name in names:
        if isinstance(value, dict) and name in value:
            return value[name]
        candidate = getattr(value, name, None)
        if candidate is not None:
            return candidate
    return None


def _commit_sha(commit: object) -> str:
    value = (
        commit if isinstance(commit, str) else _value(commit, "commit_id", "oid", "sha")
    )
    if not isinstance(value, str) or not _COMMIT_SHA.fullmatch(value):
        raise VerificationError("Hub did not return a complete immutable commit SHA")
    return value


class VerificationError(PublicationError):
    """Raised when the Hub does not contain the expected bytes."""


def _mutate_commit(
    api: HubClient,
    repo_id: str,
    *,
    operations: c.Sequence[UploadOperation],
    message: str,
    commit_recorded: c.Callable[[str], None] | None = None,
) -> object:
    _assert_private(api.repo_info(repo_id=repo_id, repo_type="dataset"), repo_id)
    result = api.create_commit(
        repo_id, operations, repo_type="dataset", commit_message=message
    )
    commit_id = _commit_sha(result)
    if commit_recorded is not None:
        commit_recorded(commit_id)
    _assert_private(api.repo_info(repo_id=repo_id, repo_type="dataset"), repo_id)
    return result


def publish_batch(
    api: HubClient,
    repo_id: str,
    batch_id: str,
    shards: c.Sequence[LocalShard],
    *,
    programme_count: int = 0,
    rejection_counts: dict[RejectionCategory, int] | None = None,
    validator: c.Callable[[object, str], None] | None = None,
    schema_validator: c.Callable[[object, str], None] | None = None,
    expected_schema: object | None = None,
    durable_verification: c.Callable[[BatchEvidence], None] | None = None,
    purge_callback: c.Callable[[tuple[Path, ...]], None] | None = None,
    staging_dir: Path | None = None,
    ledger: Ledger | None = None,
    commit_recorded: c.Callable[[str], None] | None = None,
) -> BatchEvidence:
    """Commit, verify, stream-decode, and optionally purge one bounded batch.

    Args:
        api:
            Injectable Hub client.
        repo_id:
            Private dataset repository identifier.
        batch_id:
            Stable local batch identifier.
        shards:
            Explicit local Parquet files in this batch.
        programme_count (optional):
            Number of source programmes represented by the batch.
        rejection_counts (optional):
            Metadata-only rejection counts.
        validator (optional):
            Callback invoked once for a deterministic streaming sample from every
            committed shard. The default checks that each shard is non-empty.
        durable_verification (optional):
            Callback that durably records verified evidence before any purge.
        purge_callback (optional):
            Callback that removes local artefacts after durable verification.
        staging_dir (optional):
            Durable directory for the pending manifest. Defaults to the first
            shard's directory; it is retained until the purge callback runs.
        ledger (optional):
            Ledger that records the commit before verification and verification
            before purge.
        commit_recorded (optional):
            Callback invoked immediately after the Hub returns its commit SHA.
        schema_validator (optional):
            Callback that checks the exact remote dataset schema.
        expected_schema (optional):
            Feature or Arrow schema compared exactly with each remote shard.

    Returns:
        Verified or purged batch evidence.

    Raises:
        AllowListError:
            If a local file or batch size is unsafe.
    """
    if not shards:
        raise AllowListError("a publication batch must contain at least one shard")
    if len(shards) + 1 >= 100:
        raise AllowListError("a Hub commit must contain fewer than 100 operations")
    _assert_private(api.repo_info(repo_id=repo_id, repo_type="dataset"), repo_id)
    local_evidence = tuple(_local_evidence(shard) for shard in shards)
    _assert_unique_paths(local_evidence)
    _refuse_remote_collisions(api, repo_id, local_evidence)
    counts = rejection_counts if rejection_counts is not None else {}
    _assert_safe_metadata(counts)

    manifest_root = staging_dir or shards[0].path.parent
    if not manifest_root.is_dir() or manifest_root.is_symlink():
        raise AllowListError("batch staging directory is not a regular directory")
    manifest_path = manifest_root / "batch-manifest.json"
    manifest = _manifest_bytes(
        batch_id=batch_id,
        shards=local_evidence,
        programme_count=programme_count,
        rejection_counts=counts,
    )
    if manifest_path.exists():
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise AllowListError("batch manifest is not a regular non-symlink file")
        if manifest_path.read_bytes() != manifest:
            raise AllowListError("existing batch manifest does not match this batch")
    else:
        _write_durable(manifest_path, manifest)
    manifest_evidence = ShardEvidence(
        path="batch-manifest.json",
        byte_size=len(manifest),
        row_count=0,
        sha256=_sha256_bytes(manifest),
    )
    if ledger is not None and ledger.batch(batch_id).state in {
        LedgerState.COMMITTED,
        LedgerState.VERIFIED,
        LedgerState.PURGED,
    }:
        return verify_batch(
            api,
            repo_id,
            batch_id,
            local_evidence=local_evidence,
            manifest_path=manifest_path,
            ledger=ledger,
            validator=validator,
            schema_validator=schema_validator,
            expected_schema=expected_schema,
            purge_callback=purge_callback,
            local_paths=tuple(shard.path for shard in shards),
        )
    operations = tuple(
        [
            UploadOperation(path_in_repo=item.path, path=shard.path)
            for item, shard in zip(local_evidence, shards)
        ]
        + [UploadOperation(path_in_repo=manifest_evidence.path, path=manifest_path)]
    )
    commit_id: str | None = None

    def remember_commit(value: str) -> None:
        nonlocal commit_id
        commit_id = value
        if ledger is not None:
            ledger.record_commit(batch_id, value)
        if commit_recorded is not None:
            commit_recorded(value)

    _mutate_commit(
        api,
        repo_id,
        operations=operations,
        message=f"Publish P1 batch {batch_id}",
        commit_recorded=remember_commit,
    )
    assert commit_id is not None
    return verify_batch(
        api,
        repo_id,
        batch_id,
        local_evidence=local_evidence,
        manifest_path=manifest_path,
        commit_id=commit_id,
        programme_count=programme_count,
        rejection_counts=counts,
        validator=validator,
        schema_validator=schema_validator,
        expected_schema=expected_schema,
        ledger=ledger,
        durable_verification=durable_verification,
        purge_callback=(
            None if purge_callback is None else lambda paths: purge_callback(paths)
        ),
        local_paths=tuple(shard.path for shard in shards),
    )


def verify_batch(
    api: HubClient,
    repo_id: str,
    batch_id: str,
    *,
    local_evidence: c.Sequence[ShardEvidence] | None = None,
    manifest_path: Path | None = None,
    commit_id: str | None = None,
    programme_count: int = 0,
    rejection_counts: dict[RejectionCategory, int] | None = None,
    validator: c.Callable[[object, str], None] | None = None,
    schema_validator: c.Callable[[object, str], None] | None = None,
    expected_schema: object | None = None,
    ledger: Ledger | None = None,
    durable_verification: c.Callable[[BatchEvidence], None] | None = None,
    purge_callback: c.Callable[[tuple[Path, ...]], None] | None = None,
    local_paths: c.Sequence[Path] | None = None,
) -> BatchEvidence:
    """Idempotently verify a committed batch and optionally purge its files.

    Recovery uses the immutable commit and evidence in ``ledger`` when local
    assembly state is unavailable. Every expected object is checked by path, size,
    and SHA-256 (streaming when Hub metadata does not expose a digest); the
    manifest is parsed and compared byte-for-byte to its canonical schema, then a
    deterministic streaming decode is performed for every shard. Verification is
    recorded before ``purge_callback`` is called.

    Returns:
        Complete verified batch evidence, or purged evidence when requested.

    Raises:
        PublicationError:
            If no durable commit or verification record is available.
        VerificationError:
            If any remote object, manifest, schema, or audio decode is invalid.
    """
    if ledger is not None:
        record = ledger.batch(batch_id)
        if record.commit_id is None:
            raise PublicationError("batch has no durable commit to recover")
        if commit_id is not None and commit_id != record.commit_id:
            raise PublicationError("recovery commit differs from the ledger")
        commit_id = record.commit_id
        if local_evidence is None:
            local_evidence = tuple(
                ShardEvidence(
                    path=item.path,
                    byte_size=item.byte_size,
                    row_count=item.row_count,
                    sha256=item.sha256,
                )
                for item in ledger.shards(batch_id)
            )
        programme_count = record.programme_count
        rejection_counts = {
            RejectionCategory(key): value
            for key, value in record.rejection_counts.items()
        }
    if commit_id is None or not _COMMIT_SHA.fullmatch(commit_id):
        raise PublicationError("recovery requires a complete immutable commit SHA")
    evidence_items = tuple(local_evidence or ())
    if not evidence_items:
        raise PublicationError("a publication batch must contain at least one shard")
    _assert_unique_paths(evidence_items)
    counts = rejection_counts if rejection_counts is not None else {}
    _assert_safe_metadata(counts)
    manifest = _manifest_bytes(
        batch_id=batch_id,
        shards=evidence_items,
        programme_count=programme_count,
        rejection_counts=counts,
    )
    expected = (
        *evidence_items,
        ShardEvidence(
            path="batch-manifest.json",
            byte_size=len(manifest),
            row_count=0,
            sha256=_sha256_bytes(manifest),
        ),
    )
    _assert_private(
        api.repo_info(repo_id=repo_id, repo_type="dataset", revision=commit_id), repo_id
    )
    streamed_files = _verify_remote(api, repo_id, expected=expected, revision=commit_id)
    remote_manifest = streamed_files.get("batch-manifest.json", manifest)
    if remote_manifest != manifest:
        raise VerificationError("batch manifest schema or contents differ")
    try:
        parsed = json.loads(remote_manifest)
    except (TypeError, ValueError) as error:
        raise VerificationError("batch manifest is not valid JSON") from error
    expected_payload = json.loads(manifest)
    if parsed != expected_payload or set(parsed) != set(expected_payload):
        raise VerificationError("batch manifest schema is not exact")

    check = validator or validate_streaming_sample
    for item in evidence_items:
        dataset = api.load_dataset(
            repo_id, shard_path=item.path, revision=commit_id, streaming=True
        )
        check(dataset, item.path)
        if schema_validator is not None:
            schema_validator(dataset, item.path)
        if expected_schema is not None:
            actual_schema = getattr(dataset, "features", None)
            if actual_schema is None:
                actual_schema = getattr(dataset, "schema", None)
            if actual_schema != expected_schema:
                raise VerificationError(f"exact schema mismatch for {item.path}")
        else:
            _validate_default_schema(dataset, item.path)
    result = BatchEvidence(
        batch_id=batch_id,
        state=LedgerState.VERIFIED,
        shards=evidence_items,
        commit_id=commit_id,
        programme_count=programme_count,
        row_count=sum(item.row_count for item in evidence_items),
        rejection_counts=counts,
    )
    durable = durable_verification is not None
    if ledger is not None:
        ledger.mark_batch_verified(batch_id, commit_id=commit_id, shards=evidence_items)
        for shard in ledger.shards(batch_id):
            if shard.state not in {LedgerState.VERIFIED, LedgerState.PURGED}:
                if shard.state is not LedgerState.COMMITTED:
                    ledger.transition_shard(shard.shard_id, LedgerState.COMMITTED)
                ledger.transition_shard(shard.shard_id, LedgerState.VERIFIED)
        durable = True
    if durable_verification is not None:
        durable_verification(result)
    if purge_callback is not None:
        if not durable:
            raise PublicationError("purge requires a durable verification callback")
        if ledger is not None and ledger.batch(batch_id).state is LedgerState.PURGED:
            return result.model_copy(update={"state": LedgerState.PURGED})
        paths = tuple(local_paths or ())
        if manifest_path is not None:
            paths += (manifest_path,)
        purge_callback(paths)
        if ledger is not None and ledger.batch(batch_id).state is LedgerState.VERIFIED:
            ledger.purge_batch(
                batch_id, evidence={"deleted": True, "kind": "publication-artifact"}
            )
        return result.model_copy(update={"state": LedgerState.PURGED})
    return result


recover_batch = verify_batch


def _assert_unique_paths(shards: tuple[ShardEvidence, ...]) -> None:
    paths = [shard.path for shard in shards]
    if len(paths) != len(set(paths)):
        raise AllowListError("a batch contains duplicate repository paths")


class AllowListError(PublicationError):
    """Raised when an upload contains an unsafe local path."""


def _local_evidence(shard: LocalShard) -> ShardEvidence:
    if shard.row_count < 0:
        raise AllowListError(f"negative row count for {shard.repo_path}")
    _assert_repo_path(shard.repo_path)
    if not shard.repo_path.endswith(".parquet"):
        raise AllowListError(f"only Parquet shards may be uploaded: {shard.repo_path}")
    if shard.path.is_symlink() or not shard.path.is_file():
        raise AllowListError(f"shard is not a regular non-symlink file: {shard.path}")
    digest, size = _stream_local(shard.path)
    return ShardEvidence(
        path=shard.repo_path, byte_size=size, row_count=shard.row_count, sha256=digest
    )


def _assert_repo_path(path: str) -> None:
    pure = PurePosixPath(path)
    if (
        not path
        or pure.is_absolute()
        or "\\" in path
        or ".." in pure.parts
        or str(pure) != path
    ):
        raise AllowListError(f"unsafe repository path: {path!r}")


def _stream_local(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _manifest_bytes(
    *,
    batch_id: str,
    shards: tuple[ShardEvidence, ...],
    programme_count: int,
    rejection_counts: dict[RejectionCategory, int],
) -> bytes:
    payload = {
        "batch_id": batch_id,
        "programme_count": programme_count,
        "row_count": sum(item.row_count for item in shards),
        "rejection_counts": {
            getattr(key, "value", str(key)): value
            for key, value in rejection_counts.items()
        },
        "shards": [item.model_dump(mode="json") for item in shards],
    }
    return json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode()


def _refuse_remote_collisions(
    api: HubClient, repo_id: str, expected: tuple[ShardEvidence, ...]
) -> None:
    """Refuse overwriting paths from an unrelated publication.

    Raises:
        AllowListError:
            If a requested path already exists in the repository.
    """
    listing = getattr(api, "list_repo_files", None)
    if not callable(listing):
        return
    try:
        existing = set(listing(repo_id, repo_type="dataset", revision=None))
    except Exception as error:
        raise AllowListError("could not establish remote path safety") from error
    collisions = existing.intersection(item.path for item in expected)
    if collisions:
        raise AllowListError(
            "remote publication path collision: " + ", ".join(sorted(collisions))
        )


def _stream_digest(chunks: c.Iterable[bytes]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        if not isinstance(chunk, bytes):
            raise VerificationError("remote stream yielded a non-bytes chunk")
        digest.update(chunk)
    return digest.hexdigest()


def _validate_default_schema(dataset: object, shard_path: str) -> None:
    features = getattr(dataset, "features", None)
    if features is None:
        return
    if isinstance(features, dict):
        names = tuple(features)
    else:
        keys = getattr(features, "keys", None)
        names = tuple(keys()) if callable(keys) else ()
    expected = tuple(field.name for field in OUTPUT_SCHEMA.fields)
    if names != expected:
        raise VerificationError(f"exact P1 schema mismatch for {shard_path}")


def _verify_remote(
    api: HubClient, repo_id: str, *, expected: tuple[ShardEvidence, ...], revision: str
) -> dict[str, bytes]:
    infos = tuple(
        api.get_paths_info(
            repo_id,
            [item.path for item in expected],
            repo_type="dataset",
            revision=revision,
        )
    )
    by_path = {_value(info, "path"): info for info in infos}
    if len(infos) != len(expected) or set(by_path) != {item.path for item in expected}:
        raise VerificationError("remote paths do not exactly match the manifest")
    streamed_files: dict[str, bytes] = {}
    for item in expected:
        info = by_path.get(item.path)
        if info is None:
            raise VerificationError(f"missing remote path at {revision}: {item.path}")
        size = _value(info, "size", "size_bytes")
        if not isinstance(size, int) or size != item.byte_size:
            raise VerificationError(f"size mismatch for {item.path}")
        digest = _remote_digest(info)
        if digest is None:
            content = b"".join(
                api.stream_file(
                    repo_id, item.path, repo_type="dataset", revision=revision
                )
            )
            streamed_files[item.path] = content
            digest = _sha256_bytes(content)
        if digest != item.sha256:
            raise VerificationError(f"SHA-256 mismatch for {item.path}")
    return streamed_files


def _remote_digest(info: object) -> str | None:
    lfs = _value(info, "lfs")
    digest = _value(info, "sha256", "digest") or _value(lfs, "sha256")
    if isinstance(digest, str) and _SHA256.fullmatch(digest):
        return digest
    return None


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_durable(path: Path, content: bytes) -> None:
    with path.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def validate_staging_directory(
    root: Path, allowed: c.Iterable[Path]
) -> tuple[Path, ...]:
    """Reject every staging entry not explicitly allow-listed.

    Args:
        root:
            Isolated upload staging directory.
        allowed:
            Paths relative to ``root`` that may be uploaded.

    Returns:
        The validated regular files in allow-list order.

    Raises:
        AllowListError:
            If an entry is unexpected, a symlink, or not a regular file.
    """
    allowed_paths = tuple(Path(path) for path in allowed)
    expected = set(allowed_paths)
    found = {path.relative_to(root) for path in root.rglob("*")}
    for relative in sorted(found):
        candidate = root / relative
        if candidate.is_symlink():
            raise AllowListError(f"symlink is not publishable: {relative}")
        if relative not in expected or not candidate.is_file():
            raise AllowListError(f"unexpected non-regular upload path: {relative}")
    missing = expected - found
    if missing:
        raise AllowListError(f"allow-listed path is missing: {sorted(missing)}")
    return tuple(root / path for path in allowed_paths)


def validate_streaming_sample(dataset: object, shard_path: str) -> None:
    """Decode one deterministic sample from a streaming shard dataset.

    Args:
        dataset:
            Dataset returned by ``load_dataset(..., streaming=True)``.
        shard_path:
            Remote shard path, used in the diagnostic.

    Raises:
        VerificationError:
            If the shard is empty or iteration cannot decode a sample.
    """
    try:
        sample = next(iter(t.cast(c.Iterator[object], dataset)))
    except Exception as error:
        raise VerificationError(
            f"could not decode a sample from {shard_path}"
        ) from error
    if sample is None:
        raise VerificationError(f"empty streaming shard: {shard_path}")
    if not isinstance(sample, dict) or "audio" not in sample or "text" not in sample:
        raise VerificationError(f"schema cannot decode a P1 row: {shard_path}")
    audio = sample["audio"]
    if not isinstance(audio, dict):
        raise VerificationError(f"audio is not a structured feature: {shard_path}")
    payload = audio.get("array", audio.get("bytes"))
    if payload is None or (hasattr(payload, "__len__") and len(payload) == 0):
        raise VerificationError(f"audio payload is empty: {shard_path}")
