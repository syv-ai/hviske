"""Private, verified publication of P1 dataset shards."""

from __future__ import annotations

import collections.abc as c
import hashlib
import io
import json
import os
import re
import tempfile
import typing as t
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import soundfile as sf
from datasets import Audio, Features, Sequence, Value, load_dataset
from huggingface_hub import CommitOperationAdd, HfApi, HfFileSystem, hf_hub_url
from huggingface_hub.utils import RepositoryNotFoundError
from pyarrow import parquet as pq

from .p1_contracts import (
    OUTPUT_SCHEMA,
    P1_RUNTIME_CONTRACT,
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


def _expected_features() -> Features:
    """Build the exact Hugging Face feature contract for published shards.

    Returns:
        The exact feature mapping required by ``OUTPUT_SCHEMA``.
    """
    features: dict[str, object] = {}
    for field in OUTPUT_SCHEMA.fields:
        if field.type == "Audio(16000)":
            features[field.name] = Audio(sampling_rate=16000)
        elif field.type.startswith("list"):
            features[field.name] = Sequence(Value("string"))
        else:
            features[field.name] = Value(field.type)
    return Features(features)


_EXPECTED_FEATURES = _expected_features()
_EXPECTED_ARROW_SCHEMA = _EXPECTED_FEATURES.arrow_schema
P1_FEATURES = _EXPECTED_FEATURES


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
    model_revisions: str | None = None,
    dataset_license: str | None = None,
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
        model_revisions (optional):
            Immutable model revisions for a model-backed future alignment.
            Timestamp-native cards omit this section.
        dataset_license (optional):
            Immutable target dataset licence provenance.

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
    }
    if model_revisions is not None:
        values["Model revisions"] = model_revisions
    if dataset_license is None:
        dataset_license = json.dumps(
            {
                "template_repository": (
                    P1_RUNTIME_CONTRACT.dataset_license_template_repository
                ),
                "template_revision": (
                    P1_RUNTIME_CONTRACT.dataset_license_template_revision
                ),
                "template_url": (P1_RUNTIME_CONTRACT.dataset_license_template_url),
                "template_sha256": (
                    P1_RUNTIME_CONTRACT.dataset_license_template_sha256
                ),
                "template_bytes": P1_RUNTIME_CONTRACT.dataset_license_template_bytes,
                "adaptation": P1_RUNTIME_CONTRACT.dataset_license_adaptation,
                "target_path": P1_RUNTIME_CONTRACT.dataset_license_target_path,
                "target_sha256": P1_RUNTIME_CONTRACT.dataset_license_target_sha256,
            },
            sort_keys=True,
        )
    values["Dataset licence provenance"] = dataset_license
    _assert_safe_metadata(values)
    sections = [f"## {name}\n\n{value}" for name, value in values.items()]
    return (
        "---\n"
        "license: other\n"
        "license_name: p1-dataset-license\n"
        "license_link: LICENSE\n"
        "---\n\n"
        "# P1 segmented Danish speech\n\n"
        + "\n\n".join(sections)
        + "\n\n## Licence\n\n"
        "Private access, use, and distribution are subject to [LICENSE](LICENSE).\n"
    )


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
        dataset = load_dataset(
            "parquet",
            data_files={"train": url},
            split="train",
            streaming=True,
            token=self._token,
        )
        return dataset.cast_column("audio", Audio(sampling_rate=16000, decode=False))

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
                f"datasets/{repo_id}/{path}", mode="rb", revision=revision
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
    license_text: str | None = None,
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
        license_text (optional):
            Full target dataset licence. Defaults to the tracked repository licence.
        gitattributes (optional):
            Initial Git attributes content.
        token (optional):
            Authentication token, used only to reject accidental card leakage.

    Returns:
        The immutable initialisation commit SHA, if a commit was made.

    Raises:
        PublicationError:
            If metadata is unsafe, the licence is not pinned, or the target tree is
            not pristine metadata-only v7 state.
    """
    _assert_safe_metadata(card, token=token)
    _assert_safe_metadata(gitattributes, token=token)
    if license_text is None:
        license_path = Path(__file__).resolve().parents[2] / "LICENSE-DATASET"
        license_text = license_path.read_text(encoding="utf-8")
    _assert_safe_metadata(license_text, token=token)
    if (
        hashlib.sha256(license_text.encode("utf-8")).hexdigest()
        != P1_RUNTIME_CONTRACT.dataset_license_target_sha256
    ):
        raise PublicationError("target dataset licence does not match the pinned file")
    try:
        info = api.repo_info(repo_id=repo_id, repo_type="dataset")
    except RepositoryNotFoundError:
        api.create_repo(
            repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True
        )
        info = api.repo_info(repo_id=repo_id, repo_type="dataset")
    _assert_private(info, repo_id)
    _assert_initialise_target_is_safe(api, repo_id, card)

    with tempfile.TemporaryDirectory(prefix="hviske-p1-card-") as directory:
        root = Path(directory)
        card_path = root / "README.md"
        attrs_path = root / ".gitattributes"
        license_path = root / "LICENSE"
        card_path.write_text(card, encoding="utf-8")
        attrs_path.write_text(gitattributes, encoding="utf-8")
        license_path.write_text(license_text, encoding="utf-8")
        commit = _mutate_commit(
            api,
            repo_id,
            operations=(
                UploadOperation(path_in_repo="README.md", path=card_path),
                UploadOperation(path_in_repo=".gitattributes", path=attrs_path),
                UploadOperation(path_in_repo="LICENSE", path=license_path),
            ),
            message="Initialise private P1 dataset",
        )
        _assert_private(api.repo_info(repo_id=repo_id, repo_type="dataset"), repo_id)
    return _commit_sha(commit)


def _assert_initialise_target_is_safe(api: HubClient, repo_id: str, card: str) -> None:
    """Reject target trees that could contain an earlier generation payload.

    Raises:
        PublicationError:
            If the tree cannot be inspected or contains data, unknown files, or
            incompatible metadata.
    """
    try:
        paths = tuple(
            str(path)
            for path in api.list_repo_files(
                repo_id=repo_id, repo_type="dataset", revision="main"
            )
        )
    except Exception as error:
        raise PublicationError(
            "cannot inspect the private target tree before initialisation"
        ) from error

    allowed_metadata = {"README.md", ".gitattributes", "LICENSE"}
    unexpected = sorted(set(paths) - allowed_metadata)
    if unexpected:
        raise PublicationError(
            "refusing to initialise a target containing data or unknown payload: "
            + ", ".join(unexpected)
        )
    if "README.md" not in paths:
        if "LICENSE" in paths:
            try:
                existing_license = b"".join(
                    api.stream_file(
                        repo_id, "LICENSE", repo_type="dataset", revision="main"
                    )
                )
            except Exception as error:
                raise PublicationError(
                    "cannot inspect the existing target licence before initialisation"
                ) from error
            if hashlib.sha256(existing_license).hexdigest() != (
                P1_RUNTIME_CONTRACT.dataset_license_target_sha256
            ):
                raise PublicationError(
                    "existing target licence is not the pinned dataset licence"
                )
        return
    try:
        existing_card = b"".join(
            api.stream_file(repo_id, "README.md", repo_type="dataset", revision="main")
        ).decode("utf-8")
    except Exception as error:
        raise PublicationError(
            "cannot inspect the existing target card before initialisation"
        ) from error
    required = (
        "pipeline_version: p1-segmentation-7",
        "timestamp-native:p1-transcripts.words",
        "p1-segments-v2",
    )
    if not all(marker in existing_card for marker in required):
        raise PublicationError(
            "refusing to overwrite a target card without the compatible v7 identity"
        )
    expected_digest = re.search(r"pipeline_config_sha256: ([0-9a-f]{64})", card)
    actual_digest = re.search(r"pipeline_config_sha256: ([0-9a-f]{64})", existing_card)
    if expected_digest is not None and (
        actual_digest is None or actual_digest.group(1) != expected_digest.group(1)
    ):
        raise PublicationError("existing target card has a different v7 identity")
    if re.search(r"\b(?:vad|ctc|roest|whisper|silero)\b", existing_card, re.I):
        raise PublicationError(
            "existing target card contains inactive model provenance"
        )
    if "LICENSE" in paths:
        try:
            existing_license = b"".join(
                api.stream_file(
                    repo_id, "LICENSE", repo_type="dataset", revision="main"
                )
            )
        except Exception as error:
            raise PublicationError(
                "cannot inspect the existing target licence before initialisation"
            ) from error
        if hashlib.sha256(existing_license).hexdigest() != (
            P1_RUNTIME_CONTRACT.dataset_license_target_sha256
        ):
            raise PublicationError(
                "existing target licence is not the pinned dataset licence"
            )


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
    value = _value(commit, "commit_id", "oid", "sha")
    if value is None and type(commit) is str:
        value = commit
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
        PublicationError:
            If durable ledger evidence does not match the local batch.
    """
    if shards and len(shards) + 1 >= 100:
        raise AllowListError("a Hub commit must contain fewer than 100 operations")
    _assert_private(api.repo_info(repo_id=repo_id, repo_type="dataset"), repo_id)
    ledger_record = None if ledger is None else ledger.batch(batch_id)
    if ledger_record is not None and ledger_record.state in {
        LedgerState.COMMITTED,
        LedgerState.VERIFIED,
        LedgerState.PURGED,
    }:
        # A committed batch is recovered from its immutable ledger evidence. In
        # particular, do not inspect or re-upload local shards before verifying it.
        records = ledger.shards(batch_id)
        manifest_path = next(
            (
                Path(record.local_path).parent / _manifest_repo_path(batch_id)
                for record in records
                if record.local_path is not None
            ),
            None,
        )
        local_paths = tuple(
            Path(record.local_path)
            for record in records
            if record.local_path is not None
        )
        return verify_batch(
            api,
            repo_id,
            batch_id,
            ledger=ledger,
            validator=validator,
            schema_validator=schema_validator,
            expected_schema=expected_schema,
            purge_callback=purge_callback,
            manifest_path=manifest_path,
            local_paths=local_paths,
        )
    if not shards:
        raise AllowListError("a publication batch must contain at least one shard")
    local_evidence = tuple(_local_evidence(shard) for shard in shards)
    _assert_unique_paths(local_evidence)
    if ledger_record is not None and ledger_record.state is LedgerState.SHARDED:
        durable_evidence = tuple(
            ShardEvidence(
                path=record.path,
                byte_size=record.byte_size,
                row_count=record.row_count,
                sha256=record.sha256,
            )
            for record in ledger.shards(batch_id)
        )
        if local_evidence != durable_evidence:
            raise PublicationError(
                "local shard bytes differ from durable ledger evidence"
            )
    counts = rejection_counts if rejection_counts is not None else {}
    _assert_safe_metadata(counts)

    manifest_root = staging_dir or shards[0].path.parent
    if not manifest_root.is_dir() or manifest_root.is_symlink():
        raise AllowListError("batch staging directory is not a regular directory")
    manifest_relative_path = _manifest_repo_path(batch_id)
    manifest_path = manifest_root / manifest_relative_path
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.parent.is_symlink():
        raise AllowListError("batch manifest directory must not be a symlink")
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
        path=manifest_relative_path,
        byte_size=len(manifest),
        row_count=0,
        sha256=_sha256_bytes(manifest),
    )
    _refuse_remote_collisions(api, repo_id, (*local_evidence, manifest_evidence))
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
            path=_manifest_repo_path(batch_id),
            byte_size=len(manifest),
            row_count=0,
            sha256=_sha256_bytes(manifest),
        ),
    )
    _assert_private(
        api.repo_info(repo_id=repo_id, repo_type="dataset", revision=commit_id), repo_id
    )
    streamed_files = _verify_remote(api, repo_id, expected=expected, revision=commit_id)
    remote_manifest = streamed_files.get(_manifest_repo_path(batch_id), manifest)
    if remote_manifest != manifest:
        raise VerificationError("batch manifest schema or contents differ")
    try:
        parsed = json.loads(remote_manifest)
    except (TypeError, ValueError) as error:
        raise VerificationError("batch manifest is not valid JSON") from error
    expected_payload = json.loads(manifest)
    if parsed != expected_payload or set(parsed) != set(expected_payload):
        raise VerificationError("batch manifest schema is not exact")

    for item in evidence_items:
        dataset = api.load_dataset(
            repo_id, shard_path=item.path, revision=commit_id, streaming=True
        )
        actual_schema = getattr(dataset, "features", None)
        if actual_schema is None:
            actual_schema = getattr(dataset, "schema", None)
        if expected_schema is not None:
            _assert_exact_schema(actual_schema, expected_schema, item.path)
        elif actual_schema is not None:
            _assert_exact_schema(actual_schema, _EXPECTED_FEATURES, item.path)
        if actual_schema is not None or validator is None:
            validate_streaming_sample(dataset, item.path)
        if validator is not None:
            validator(dataset, item.path)
        if schema_validator is not None:
            schema_validator(dataset, item.path)
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


def _has_parquet_footer(path: Path) -> bool:
    """Identify a Parquet candidate without reading its payload into memory.

    Returns:
        Whether the file has the Parquet magic bytes at both ends.
    """
    if path.stat().st_size < 8:
        return False
    with path.open("rb") as stream:
        header = stream.read(4)
        stream.seek(-4, os.SEEK_END)
        footer = stream.read(4)
    return header == b"PAR1" and footer == b"PAR1"


def _local_evidence(shard: LocalShard) -> ShardEvidence:
    if shard.row_count < 0:
        raise AllowListError(f"negative row count for {shard.repo_path}")
    _assert_repo_path(shard.repo_path)
    if not shard.repo_path.endswith(".parquet"):
        raise AllowListError(f"only Parquet shards may be uploaded: {shard.repo_path}")
    if shard.path.is_symlink() or not shard.path.is_file():
        raise AllowListError(f"shard is not a regular non-symlink file: {shard.path}")
    validate_local_shard(shard.path, expected_row_count=shard.row_count)
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


def validate_local_shard(path: Path, *, expected_row_count: int | None = None) -> None:
    """Validate one local shard before it is eligible for upload.

    The Parquet Arrow schema and Hugging Face feature metadata are compared as one
    object, so missing, extra, and type-wrong fields cannot pass. Every row is
    decoded to prove the mono 16 kHz FLAC contract without retaining a complete
    shard in memory.

    Args:
        path:
            Local Parquet shard.
        expected_row_count (optional):
            Row count recorded by the assembler. Defaults to no count check.

    Raises:
        VerificationError:
            If the schema, row count, audio payload, digest, or duration is invalid.
    """
    if path.is_symlink() or not path.is_file():
        raise VerificationError(f"local shard is not a regular file: {path}")
    try:
        parquet = pq.ParquetFile(path)
        if parquet.schema_arrow != _EXPECTED_ARROW_SCHEMA:
            raise VerificationError(f"exact P1 schema mismatch for {path}")
        if expected_row_count is not None and parquet.metadata is not None:
            if parquet.metadata.num_rows != expected_row_count:
                raise VerificationError(f"row count mismatch for {path}")
        batches = parquet.iter_batches(batch_size=1)
        batch = next(batches, None)
        if batch is None or batch.num_rows == 0:
            raise VerificationError(f"empty local shard: {path}")
        _validate_row(batch.to_pylist()[0], str(path))
        for batch in batches:
            for row in batch.to_pylist():
                _validate_row(row, str(path))
    except VerificationError:
        raise
    except Exception as error:
        raise VerificationError(f"could not read local shard: {path}") from error
    return


def _validate_row(row: object, shard_path: str) -> None:
    if not isinstance(row, dict):
        raise VerificationError(f"decoded row is not a mapping: {shard_path}")
    expected_names = {field.name for field in OUTPUT_SCHEMA.fields}
    if set(row) != expected_names:
        raise VerificationError(f"exact P1 fields mismatch for {shard_path}")
    audio = row.get("audio")
    if not isinstance(audio, dict):
        raise VerificationError(f"audio is not a structured feature: {shard_path}")
    payload = audio.get("bytes")
    if not isinstance(payload, bytes) or not payload:
        raise VerificationError(f"audio payload is not embedded FLAC: {shard_path}")
    try:
        decoded, sample_rate = sf.read(
            io.BytesIO(payload), dtype="float32", always_2d=True
        )
    except Exception as error:
        raise VerificationError(
            f"audio payload cannot be decoded: {shard_path}"
        ) from error
    if sample_rate != 16000 or decoded.shape[1] != 1:
        raise VerificationError(f"audio is not mono 16 kHz: {shard_path}")
    if hashlib.sha256(payload).hexdigest() != row.get("audio_sha256"):
        raise VerificationError(f"audio payload digest mismatch: {shard_path}")
    duration = row.get("duration_ms")
    source_start = row.get("source_start_ms")
    source_end = row.get("source_end_ms")
    source_duration = row.get("source_duration_ms")
    if (
        not isinstance(duration, int)
        or not isinstance(source_start, int)
        or not isinstance(source_duration, int)
    ):
        raise VerificationError(f"audio duration metadata is invalid: {shard_path}")
    if (
        not isinstance(source_end, int)
        or source_end - source_start != duration
        or source_end > source_duration
    ):
        raise VerificationError(f"source duration is inconsistent: {shard_path}")
    if decoded.shape[0] != duration * 16:
        raise VerificationError(f"decoded duration is inconsistent: {shard_path}")
    if row.get("pipeline_version") == "p1-segmentation-7":
        if row.get("alignment_method") != "timestamp-native:p1-transcripts.words":
            raise VerificationError(
                f"v7 row does not declare timestamp-native alignment: {shard_path}"
            )
        if row.get("alignment_backend") != "timestamp-native":
            raise VerificationError(f"v7 row has a non-native backend: {shard_path}")
        if row.get("alignment_score_type") != "not_applicable:source_timestamps":
            raise VerificationError(
                f"v7 row has an applicable score type: {shard_path}"
            )
        if (
            row.get("source_start_ms") != row.get("proposal_start_ms")
            or row.get("source_end_ms") != row.get("proposal_end_ms")
            or row.get("alignment_score") is not None
            or row.get("start_drift_ms") is not None
            or row.get("end_drift_ms") is not None
            or row.get("vad_speech_ratio") is not None
        ):
            raise VerificationError(
                f"v7 row contains non-native alignment evidence: {shard_path}"
            )


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


def _manifest_repo_path(batch_id: str) -> str:
    """Return the immutable repository path for one batch manifest.

    Raises:
        AllowListError:
            If the batch identifier cannot safely be used as a path component.
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", batch_id):
        raise AllowListError("batch_id is not safe for a manifest path")
    return f"manifests/{batch_id}.json"


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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _stream_digest(chunks: c.Iterable[bytes]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        if not isinstance(chunk, bytes):
            raise VerificationError("remote stream yielded a non-bytes chunk")
        digest.update(chunk)
    return digest.hexdigest()


def _validate_default_schema(dataset: object, shard_path: str) -> None:
    """Validate the default feature schema when a loader exposes one."""
    features = getattr(dataset, "features", None)
    if features is None:
        return
    _assert_exact_schema(features, _EXPECTED_FEATURES, shard_path)


def _assert_exact_schema(actual: object, expected: object, shard_path: str) -> None:
    """Reject any schema that is not exactly the requested Arrow/features schema.

    Raises:
        VerificationError:
            If either schema is absent or differs from the expected schema.
    """
    if actual is None:
        raise VerificationError(f"remote dataset has no schema: {shard_path}")
    try:
        actual_features = _as_features(actual)
        expected_features = _as_features(expected)
        if actual_features is not None and expected_features is not None:
            if set(actual_features) != set(expected_features):
                raise VerificationError(f"exact P1 fields mismatch for {shard_path}")
            for name, expected_feature in expected_features.items():
                actual_feature = actual_features[name]
                if isinstance(expected_feature, Audio) and isinstance(
                    actual_feature, Audio
                ):
                    if (
                        actual_feature.sampling_rate != expected_feature.sampling_rate
                        or actual_feature.mono != expected_feature.mono
                    ):
                        raise VerificationError(
                            f"exact audio feature mismatch for {shard_path}"
                        )
                elif actual_feature != expected_feature:
                    raise VerificationError(
                        f"exact P1 feature mismatch for {shard_path}"
                    )
        actual_arrow = _as_arrow_schema(actual)
        expected_arrow = _as_arrow_schema(expected)
    except VerificationError:
        raise
    except Exception as error:
        raise VerificationError(f"invalid P1 schema for {shard_path}") from error
    if actual_arrow != expected_arrow:
        raise VerificationError(f"exact P1 schema mismatch for {shard_path}")


def _as_arrow_schema(value: object) -> object:
    """Normalise a Datasets feature mapping or Arrow schema for comparison.

    Returns:
        The Arrow schema represented by ``value``.
    """
    if isinstance(value, dict):
        return Features(value).arrow_schema
    return getattr(value, "arrow_schema", value)


def _as_features(value: object) -> Features | None:
    """Normalise feature mappings for semantic feature comparisons.

    Returns:
        Normalised features, or None for an Arrow-only schema.
    """
    if isinstance(value, Features):
        return value
    if isinstance(value, dict):
        return Features(value)
    return None


def _verify_remote(
    api: HubClient, repo_id: str, *, expected: tuple[ShardEvidence, ...], revision: str
) -> dict[str, bytes]:
    """Verify remote objects while retaining only the small manifest.

    Returns:
        The retained manifest bytes, if the remote metadata did not expose a digest.

    Raises:
        VerificationError:
            If remote paths, sizes, or digests differ from the expected evidence.
    """
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
    retained: dict[str, bytes] = {}
    for item in expected:
        info = by_path.get(item.path)
        if info is None:
            raise VerificationError(f"missing remote path at {revision}: {item.path}")
        size = _value(info, "size", "size_bytes")
        if not isinstance(size, int) or size != item.byte_size:
            raise VerificationError(f"size mismatch for {item.path}")
        digest = _remote_digest(info)
        if digest is None:
            digest, content = _stream_remote(
                api,
                repo_id,
                item=item,
                revision=revision,
                retain=item.path.startswith("manifests/")
                and item.path.endswith(".json"),
            )
            if content is not None:
                retained[item.path] = content
        if digest != item.sha256:
            raise VerificationError(f"SHA-256 mismatch for {item.path}")
    return retained


def _remote_digest(info: object) -> str | None:
    lfs = _value(info, "lfs")
    digest = _value(info, "sha256", "digest") or _value(lfs, "sha256")
    if isinstance(digest, str) and _SHA256.fullmatch(digest):
        return digest
    return None


def _stream_remote(
    api: HubClient, repo_id: str, *, item: ShardEvidence, revision: str, retain: bool
) -> tuple[str, bytes | None]:
    """Hash remote chunks directly, optionally retaining a bounded manifest.

    Returns:
        The calculated digest and optional manifest bytes.

    Raises:
        VerificationError:
            If a stream yields invalid chunks or the declared size is not met.
    """
    digest = hashlib.sha256()
    content = bytearray() if retain else None
    total = 0
    for chunk in api.stream_file(
        repo_id, item.path, repo_type="dataset", revision=revision
    ):
        if not isinstance(chunk, bytes):
            raise VerificationError(f"remote stream yielded non-bytes: {item.path}")
        total += len(chunk)
        if total > item.byte_size:
            raise VerificationError(f"remote stream exceeds declared size: {item.path}")
        digest.update(chunk)
        if content is not None:
            content.extend(chunk)
    if total != item.byte_size:
        raise VerificationError(f"remote stream ended at the wrong size: {item.path}")
    return digest.hexdigest(), None if content is None else bytes(content)


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
    if not isinstance(sample, dict):
        raise VerificationError(f"schema cannot decode a P1 row: {shard_path}")
    if hasattr(sample, "keys") and set(sample) not in (
        {field.name for field in OUTPUT_SCHEMA.fields},
        {"audio", "text"},
    ):
        raise VerificationError(f"exact P1 fields mismatch for {shard_path}")
    audio = sample.get("audio")
    if not isinstance(audio, dict):
        raise VerificationError(f"audio is not a structured feature: {shard_path}")
    payload = audio.get("bytes")
    if isinstance(payload, bytes):
        _validate_row(sample, shard_path)
        return
    legacy_sample = set(sample) == {"audio", "text"}
    # A test double or a loader configured with decode=True may expose the decoded
    # array only.  It can prove shape and rate, but never the encoded-payload hash.
    array = audio.get("array")
    if legacy_sample:
        if array is None or (hasattr(array, "__len__") and len(array) == 0):
            raise VerificationError(f"audio payload is empty: {shard_path}")
        return
    sample_rate = audio.get("sampling_rate")
    if array is None or sample_rate != 16000:
        raise VerificationError(
            f"audio is not a decodable 16 kHz feature: {shard_path}"
        )
    shape = getattr(array, "shape", None)
    if shape is not None and (len(shape) != 1 or shape[0] == 0):
        raise VerificationError(f"audio is not mono: {shard_path}")
    if hasattr(array, "__len__") and len(array) == 0:
        raise VerificationError(f"audio payload is empty: {shard_path}")
