"""Focused tests for private, verified P1 publication."""

from __future__ import annotations

import collections.abc as c
import hashlib
import io
import json
import tempfile
import typing as t
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf
import yaml
from huggingface_hub import CommitInfo, HfApi, HfFileSystem
from huggingface_hub.errors import BadRequestError
from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError

from p1_dataset.contracts import LedgerState, OutputRow, ShardEvidence
from p1_dataset.hub_diagnostics import classify_hub_error
from p1_dataset.ledger import Ledger
from p1_dataset.publish import (
    AllowListError,
    HfApiAdapter,
    LocalShard,
    PrivacyError,
    PublicationError,
    UploadOperation,
    VerificationError,
    _commit_sha,
    _stream_remote,
    build_dataset_card,
    initialise_private_dataset,
    publish_batch,
    validate_local_shard,
    validate_staging_directory,
    verify_batch,
)
from p1_dataset.segments import _rows_table


def test_batch_verifies_every_path_and_streams_every_shard(tmp_path: Path) -> None:
    """Every shard is checked remotely and opened in streaming mode."""
    first = tmp_path / "one.parquet"
    second = tmp_path / "two.parquet"
    write_valid_shard(first)
    write_valid_shard(second)
    hub = MemoryHub()
    samples: list[str] = []

    def validate(dataset: object, path: str) -> None:
        """Decode the fake sample and record the shard path."""
        next(iter(t.cast(c.Iterable[object], dataset)))
        samples.append(path)

    evidence = publish_batch(
        hub,
        "org/p1",
        "batch-001",
        [
            LocalShard(first, "shards/one.parquet", 1),
            LocalShard(second, "shards/two.parquet", 1),
        ],
        expected_pipeline_version="test",
        expected_pipeline_config_sha256="b" * 64,
        validator=validate,
    )

    assert evidence.commit_id == "a" * 40
    assert evidence.row_count == 2
    assert samples == ["shards/one.parquet", "shards/two.parquet"]
    assert hub.streamed == [
        "shards/one.parquet",
        "shards/two.parquet",
        "manifests/batch-001.json",
    ]
    assert hub.loaded == [("shards/one.parquet", True), ("shards/two.parquet", True)]
    assert first.exists() and second.exists()


@dataclass
class MemoryHub:
    """Small in-memory Hub fake that records every operation."""

    private: object = True
    sha: str | None = None
    commit_id: str = "a" * 40
    expose_digest: bool = False
    flip_public: bool = False
    decode_empty: bool = False
    missing: bool = False
    corrupt_stream: bool = False
    streaming_override: tuple[str, object] | None = None
    existing_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Initialise the fake's mutable repository state."""
        self.files: dict[str, bytes] = {}
        self.commits: list[tuple[str, ...]] = []
        self.parent_commits: list[str | None] = []
        self.privacy_checks = 0
        self.streamed: list[str] = []
        self.loaded: list[tuple[str, bool]] = []
        self.created = False

    def create_commit(
        self,
        repo_id: str,
        operations: c.Iterable[UploadOperation],
        *,
        repo_type: str,
        commit_message: str,
        parent_commit: str | None = None,
    ) -> object:
        """Copy uploaded operation bytes into the fake repository.

        Returns:
            Fake commit metadata.
        """
        operations = tuple(operations)
        self.parent_commits.append(parent_commit)
        self.commits.append(tuple(operation.path_in_repo for operation in operations))
        for operation in operations:
            self.files[operation.path_in_repo] = operation.path.read_bytes()
        if self.flip_public:
            self.private = False
        return SimpleNamespace(commit_id=self.commit_id)

    def create_repo(
        self, repo_id: str, *, repo_type: str, private: bool, exist_ok: bool
    ) -> object:
        """Create the fake repository with the requested visibility.

        Returns:
            Fake repository metadata.
        """
        self.private = private
        self.created = True
        return SimpleNamespace(private=private)

    def get_paths_info(
        self, repo_id: str, paths: list[str], *, repo_type: str, revision: str
    ) -> list[object]:
        """Return size and optionally content-digest metadata."""
        result = []
        for path in paths:
            if path not in self.files and path not in self.existing_paths:
                continue
            content = self.files.get(path, b"")
            attrs: dict[str, object] = {"path": path, "size": len(content)}
            if self.expose_digest:
                attrs["sha256"] = hashlib.sha256(content).hexdigest()
            result.append(SimpleNamespace(**attrs))
        return result

    def list_repo_files(
        self, repo_id: str, *, repo_type: str, revision: str | None = None
    ) -> c.Iterable[str]:
        """Return paths already present in the fake repository."""
        return self.existing_paths

    def load_dataset(
        self, repo_id: str, *, shard_path: str, revision: str, streaming: bool
    ) -> object:
        """Return a one-row streaming dataset."""
        self.loaded.append((shard_path, streaming))
        if self.decode_empty:
            return []
        rows = pq.read_table(io.BytesIO(self.files[shard_path])).to_pylist()
        if self.streaming_override is not None:
            field, value = self.streaming_override
            rows[0][field] = value
        return rows

    def repo_info(
        self, repo_id: str, *, repo_type: str, revision: str | None = None
    ) -> object:
        """Return the current fake visibility.

        Raises:
            RepositoryNotFoundError:
                If this fake is configured as an uncreated repository.
        """
        self.privacy_checks += 1
        if self.missing and not self.created:
            raise RepositoryNotFoundError(
                "missing",
                response=httpx.Response(
                    status_code=404, request=httpx.Request("GET", "https://hub.test")
                ),
            )
        return SimpleNamespace(private=self.private, sha=self.sha)

    def stream_file(
        self, repo_id: str, path: str, *, repo_type: str, revision: str
    ) -> list[bytes]:
        """Return the remote object in multiple chunks."""
        self.streamed.append(path)
        if self.corrupt_stream and path == "one.parquet":
            return [b"remote corruption"]
        content = self.files[path]
        return [content[:1], content[1:]]


def write_valid_shard(path: Path) -> None:
    """Write one contract-valid shard for publisher integration tests."""
    payload_stream = io.BytesIO()
    sf.write(payload_stream, np.zeros(160, dtype=np.float32), 16_000, format="FLAC")
    payload = payload_stream.getvalue()
    row = OutputRow(
        audio=payload,
        audio_sha256=hashlib.sha256(payload).hexdigest(),
        text="hej",
        alignment_text="hej",
        alignment_word_map=("hej",),
        segment_id="a" * 64,
        source_file_id="source",
        source_start_ms=0,
        source_end_ms=10,
        source_duration_ms=10,
        duration_ms=10,
        speaker_ids=(),
        proposal_start_ms=0,
        proposal_end_ms=10,
        alignment_score=1.0,
        alignment_score_type="test",
        start_drift_ms=0,
        end_drift_ms=0,
        vad_speech_ratio=1.0,
        alignment_backend="test",
        pipeline_version="test",
        pipeline_config_sha256="b" * 64,
    )
    pq.write_table(_rows_table([row]), path)


@pytest.mark.parametrize(
    ("field", "value"),
    [("pipeline_version", "future"), ("pipeline_config_sha256", "d" * 64)],
)
def test_builtin_streaming_identity_checks_run_without_schema_or_custom_validator(
    tmp_path: Path, field: str, value: object
) -> None:
    """Custom streaming validators cannot replace built-in identity checks."""
    path = tmp_path / "one.parquet"
    write_valid_shard(path)
    hub = MemoryHub(streaming_override=(field, value))

    with pytest.raises(VerificationError, match="pipeline"):
        publish_batch(
            hub,
            "org/p1",
            "batch",
            [LocalShard(path, "one.parquet", 1)],
            expected_pipeline_version="test",
            expected_pipeline_config_sha256="b" * 64,
            validator=lambda _dataset, _path: None,
        )


def test_card_contains_required_terms_and_no_credentials() -> None:
    """Cards contain the required private-use statement and reject tokens."""
    card = make_card()
    assert "No public redistribution grant" not in card
    assert "license: other" in card
    assert "license_name: syvai-layered-data-license" in card
    assert "license_link: LICENSE" in card
    assert "CC BY 4.0" in card
    assert "embedded DR audio" in card
    assert "transcript text" in card
    assert "/main/" not in card
    assert all(section in card for section in ("## Source", "## Licence"))
    assert all(term not in card for term in ("CoRal", "Alexandra", "Roest"))
    with pytest.raises(PublicationError):
        initialise_private_dataset(
            MemoryHub(), "org/p1", card="token=hf_" + "x" * 20, token="hf_" + "x" * 20
        )


def make_card() -> str:
    """Build representative card metadata for tests.

    Returns:
        A safe dataset card.
    """
    return build_dataset_card(
        source_provenance="Pinned source programmes",
        permitted_use="Internal ASR research",
        private_access_terms="Access is limited to the project organisation",
        alignment_method=(
            "pipeline_version: p1-segmentation-8; "
            "alignment_method: timestamp-native:p1-transcripts.words; "
            f"pipeline_config_sha256: {'c' * 64}; "
            "Source word timestamps are authoritative."
        ),
        field_schema="audio, text and deterministic metadata",
        known_limitations="Danish speech only",
        rejection_policy="Reject undecodable or poorly aligned material",
        source_revisions=json.dumps(
            {
                "audio": {
                    "repository": "syvai/p1",
                    "revision": "449b9c2294026df6d0d37538f279fdec03f565ff",
                },
                "transcripts": {
                    "repository": "syvai/p1-transcripts",
                    "revision": "41132579816d86e889635f84f30511279f026359",
                },
            },
            sort_keys=True,
        ),
    )


def test_card_is_readable_and_contract_driven() -> None:
    """Cards present provenance, schema, and safe loading instructions clearly."""
    card = make_card()
    metadata = yaml.safe_load(card.split("---", 2)[1])

    assert metadata == {
        "pretty_name": "DR P1 speech segments",
        "language": ["da"],
        "task_categories": ["automatic-speech-recognition"],
        "license": "other",
        "license_name": "syvai-layered-data-license",
        "license_link": "LICENSE",
    }
    assert [
        section
        for section in ("## Dataset", "## Source", "## Access", "## Licence")
        if section in card
    ] == ["## Dataset", "## Source", "## Access", "## Licence"]
    assert "roughly 2006–2022" in card
    assert "ElevenLabs Scribe v2" in card
    assert "mono 16 kHz OGG/Opus" in card
    assert "verbatim text, timing, and speaker metadata" in card
    assert "authorised users" in card
    assert 'revision="<immutable-commit-sha>"' in card
    assert "streaming=True" in card
    assert "Reproducibility details" not in card
    assert "p1-segments-v2" not in card
    assert "p1-segmentation-8" not in card
    assert "Alexandra" not in card
    assert "CoRal" not in card
    assert "Roest" not in card


def test_collision_check_queries_only_proposed_paths(tmp_path: Path) -> None:
    """Collision safety does not enumerate an increasingly large repository tree."""
    path = tmp_path / "one.parquet"
    write_valid_shard(path)

    class NoListingHub(MemoryHub):
        def list_repo_files(
            self, repo_id: str, *, repo_type: str, revision: str | None = None
        ) -> c.Iterable[str]:
            del repo_id, repo_type, revision
            raise AssertionError("full repository listing must not be used")

    hub = NoListingHub(sha="b" * 40)
    publish_batch(
        hub,
        "org/p1",
        "batch",
        [LocalShard(path, "one.parquet", 1)],
        expected_pipeline_version="test",
        expected_pipeline_config_sha256="b" * 64,
    )

    assert hub.commits == [("one.parquet", "manifests/batch.json")]


def test_commit_has_fewer_than_100_operations(tmp_path: Path) -> None:
    """A batch that would reach 100 Hub operations is refused."""
    shards = []
    for index in range(99):
        path = tmp_path / f"{index}.parquet"
        path.write_bytes(b"x")
        shards.append(LocalShard(path, f"{index}.parquet", 1))
    with pytest.raises(AllowListError):
        publish_batch(
            MemoryHub(),
            "org/p1",
            "batch",
            shards,
            expected_pipeline_version="test",
            expected_pipeline_config_sha256="b" * 64,
        )


def test_commit_is_recoverable_before_verification_and_purge(tmp_path: Path) -> None:
    """A failed verification leaves a committed ledger record for recovery."""
    path = tmp_path / "one.parquet"
    write_valid_shard(path)
    database = tmp_path / "ledger.sqlite"
    hub = MemoryHub()
    with Ledger(database) as ledger:
        ledger.register_batch("batch", pipeline_digest="a" * 64)
        ledger.register_shard(
            "shard",
            path="one.parquet",
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            byte_size=path.stat().st_size,
            row_count=1,
            batch_id="batch",
        )
        ledger.transition_batch("batch", LedgerState.PROCESSING)
        ledger.transition_batch("batch", LedgerState.SHARDED)
        with pytest.raises(VerificationError):
            publish_batch(
                hub,
                "org/p1",
                "batch",
                [LocalShard(path, "one.parquet", 1)],
                expected_pipeline_version="test",
                expected_pipeline_config_sha256="b" * 64,
                ledger=ledger,
                validator=lambda _dataset, _path: (_ for _ in ()).throw(
                    VerificationError("injected crash")
                ),
            )
        assert ledger.batch("batch").state is LedgerState.COMMITTED
    with Ledger(database) as ledger:
        verified = verify_batch(
            hub,
            "org/p1",
            "batch",
            ledger=ledger,
            manifest_path=tmp_path / "manifests" / "batch-001.json",
            expected_pipeline_version="test",
            expected_pipeline_config_sha256="b" * 64,
        )
        assert verified.state is LedgerState.VERIFIED
        assert ledger.batch("batch").state is LedgerState.VERIFIED


def test_commit_sha_prefers_commit_info_oid_and_accepts_plain_sha() -> None:
    """CommitInfo URLs do not hide their immutable object identifiers."""
    commit = CommitInfo(
        commit_url="https://huggingface.co/datasets/org/repo/commit/main",
        commit_message="",
        commit_description="",
        oid="a" * 40,
    )
    assert _commit_sha(commit) == "a" * 40
    assert _commit_sha("b" * 40) == "b" * 40


@pytest.mark.parametrize(
    "commit",
    [
        "main",
        "a" * 39,
        "a" * 41,
        CommitInfo(
            commit_url="https://huggingface.co/datasets/org/repo/commit/main",
            commit_message="",
            commit_description="",
            oid="main",
        ),
    ],
)
def test_commit_sha_rejects_non_sha_values(commit: object) -> None:
    """Branches, abbreviated SHAs, and invalid CommitInfo metadata are refused."""
    with pytest.raises(VerificationError):
        _commit_sha(commit)


def test_digest_failure_retains_local_artefacts(tmp_path: Path) -> None:
    """A digest mismatch never invokes a purge."""
    path = tmp_path / "one.parquet"
    write_valid_shard(path)
    hub = MemoryHub(corrupt_stream=True)
    with pytest.raises(VerificationError):
        publish_batch(
            hub,
            "org/p1",
            "batch",
            [LocalShard(path, "one.parquet", 1)],
            expected_pipeline_version="test",
            expected_pipeline_config_sha256="b" * 64,
        )
    assert path.exists()
    assert (tmp_path / "manifests" / "batch.json").exists()


def test_empty_repository_uses_no_head_api_semantics() -> None:
    """An empty Hub target lists at no revision and commits with no parent."""
    hub = EmptyRepoHub()
    commit = initialise_private_dataset(hub, "org/p1", card=make_card())
    assert commit == "a" * 40
    assert hub.parent_commits == [None]


class EmptyRepoHub(MemoryHub):
    """Hub fake whose empty repository has no addressable default revision."""

    def list_repo_files(
        self, repo_id: str, *, repo_type: str, revision: str | None = None
    ) -> c.Iterable[str]:
        """Expose the Hub's no-head listing behaviour.

        Raises:
            RevisionNotFoundError:
                Always, because this fake represents an empty repository.
        """
        assert revision is None
        raise RevisionNotFoundError(
            "empty repository",
            response=httpx.Response(
                status_code=404, request=httpx.Request("GET", "https://hub.test")
            ),
        )


def test_exposed_digest_avoids_remote_download() -> None:
    """An exposed SHA-256 avoids streaming Parquet payloads."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "one.parquet"
        write_valid_shard(path)
        hub = MemoryHub(expose_digest=True)
        publish_batch(
            hub,
            "org/p1",
            "batch",
            [LocalShard(path, "one.parquet", 1)],
            expected_pipeline_version="test",
            expected_pipeline_config_sha256="b" * 64,
        )
        assert not hub.streamed


def test_failed_verification_resumes_without_reupload_or_early_purge(
    tmp_path: Path,
) -> None:
    """A committed batch resumes from its manifest and immutable commit."""
    path = tmp_path / "one.parquet"
    write_valid_shard(path)
    database = tmp_path / "ledger.sqlite"
    hub = MemoryHub()
    failed = True

    with Ledger(database) as ledger:
        ledger.register_batch("batch", pipeline_digest="a" * 64)
        ledger.register_shard(
            "shard",
            path="one.parquet",
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            byte_size=path.stat().st_size,
            row_count=1,
            local_path=path,
            batch_id="batch",
        )
        ledger.transition_batch("batch", LedgerState.PROCESSING)
        ledger.transition_batch("batch", LedgerState.SHARDED)

        def validator(_dataset: object, _path: str) -> None:
            """Fail the first verification and allow the recovery attempt.

            Raises:
                VerificationError:
                    On the first invocation to exercise restart recovery.
            """
            nonlocal failed
            if failed:
                failed = False
                raise VerificationError("fail once")

        with pytest.raises(VerificationError, match="fail once"):
            publish_batch(
                hub,
                "org/p1",
                "batch",
                [LocalShard(path, "one.parquet", 1)],
                expected_pipeline_version="test",
                expected_pipeline_config_sha256="b" * 64,
                ledger=ledger,
                validator=validator,
            )
        assert ledger.batch("batch").state is LedgerState.COMMITTED
        assert ledger.batch("batch").commit_id == hub.commit_id
        assert path.exists()
        manifest_path = tmp_path / "manifests" / "batch.json"
        assert manifest_path.exists()

        def purge(paths: tuple[Path, ...]) -> None:
            """Delete only the publisher's verified artefacts."""
            for candidate in paths:
                candidate.unlink()

        evidence = publish_batch(
            hub,
            "org/p1",
            "batch",
            [LocalShard(path, "one.parquet", 1)],
            expected_pipeline_version="test",
            expected_pipeline_config_sha256="b" * 64,
            ledger=ledger,
            validator=validator,
            purge_callback=purge,
        )
        assert evidence.state is LedgerState.PURGED
        assert len(hub.commits) == 1
        assert not path.exists()
        assert not manifest_path.exists()


def test_hf_adapter_forces_lfs_and_preserves_exact_bytes(tmp_path: Path) -> None:
    """Buffered inputs avoid Xet while retaining the exact operation bytes."""
    path = tmp_path / "private-source-id.parquet"
    expected = b"exact parquet bytes"
    path.write_bytes(expected)
    adapter = object.__new__(HfApiAdapter)
    adapter._token = True

    class Api:
        def create_commit(self, **kwargs: object) -> object:
            operations = t.cast(list[object], kwargs["operations"])
            handles = [
                getattr(operation, "path_or_fileobj") for operation in operations
            ]
            assert all(isinstance(handle, io.BufferedIOBase) for handle in handles)
            assert [handle.read() for handle in handles] == [expected]
            return SimpleNamespace(commit_id="a" * 40)

    adapter._api = t.cast(HfApi, Api())
    result = adapter.create_commit(
        "org/private",
        [UploadOperation(path_in_repo="data/one.parquet", path=path)],
        repo_type="dataset",
        commit_message="test",
        parent_commit="b" * 40,
    )

    assert getattr(result, "commit_id") == "a" * 40


def test_hf_adapter_tags_opaque_create_failure_without_leaking(tmp_path: Path) -> None:
    """Adapter phase hints are bounded even when transport details are private."""
    path = tmp_path / "private-source-id.parquet"
    path.write_bytes(b"bytes")
    adapter = object.__new__(HfApiAdapter)
    adapter._token = True
    error = BadRequestError(
        "private-source-id token=hf_secret",
        response=httpx.Response(
            400,
            request=httpx.Request("POST", "https://signed.test/object?token=secret"),
        ),
    )

    class Api:
        def create_commit(self, **kwargs: object) -> object:
            del kwargs
            raise error

    adapter._api = t.cast(HfApi, Api())
    with pytest.raises(BadRequestError) as captured:
        adapter.create_commit(
            "org/private",
            [UploadOperation(path_in_repo="data/one.parquet", path=path)],
            repo_type="dataset",
            commit_message="test",
        )

    diagnostic = classify_hub_error(captured.value)
    assert diagnostic.phase == "create_commit"
    assert diagnostic.reason == "unknown"
    assert "private" not in repr(diagnostic)
    assert "signed.test" not in repr(diagnostic)
    assert "secret" not in repr(diagnostic)


def test_hf_digest_stream_uses_explicit_dataset_namespace() -> None:
    """The filesystem adapter never ambiguously addresses a model repository."""
    adapter = object.__new__(HfApiAdapter)
    paths: list[str] = []

    class Filesystem:
        def open(self, path: str, **_: object) -> io.BytesIO:
            paths.append(path)
            return io.BytesIO(b"payload")

    adapter._filesystem = t.cast(HfFileSystem, Filesystem())
    assert list(
        adapter.stream_file(
            "org/p1", "shards/one.parquet", repo_type="dataset", revision="a" * 40
        )
    ) == [b"payload"]
    assert paths == ["datasets/org/p1/shards/one.parquet"]


def test_initialisation_commits_card_and_attributes_privately() -> None:
    """Initialisation uploads only the card and Git attributes privately."""
    hub = MemoryHub()
    commit = initialise_private_dataset(hub, "org/p1", card=make_card())
    assert commit == "a" * 40
    assert hub.commits == [("README.md", ".gitattributes", "LICENSE")]
    assert "*.parquet" in hub.files[".gitattributes"].decode()
    assert hub.files["LICENSE"]
    assert hub.privacy_checks >= 3


def test_initialisation_refuses_existing_generation_payload() -> None:
    """Initialisation cannot relabel an existing payload as v8 metadata."""
    hub = MemoryHub(existing_paths=("data/v6.parquet",))

    with pytest.raises(PublicationError, match="data or unknown payload"):
        initialise_private_dataset(hub, "org/p1", card=make_card())

    assert hub.commits == []


def test_invalid_commit_is_not_accepted(tmp_path: Path) -> None:
    """Branches returned by publication cannot be captured."""
    path = tmp_path / "one.parquet"
    write_valid_shard(path)
    hub = MemoryHub(commit_id="main")
    with pytest.raises(VerificationError):
        publish_batch(
            hub,
            "org/p1",
            "batch",
            [LocalShard(path, "one.parquet", 1)],
            expected_pipeline_version="test",
            expected_pipeline_config_sha256="b" * 64,
        )
    assert path.exists()


def test_local_validation_enforces_exact_schema_audio_and_duration(
    tmp_path: Path,
) -> None:
    """Local publication rejects wrong feature metadata and audio evidence."""
    payload_stream = io.BytesIO()
    sf.write(payload_stream, np.zeros(160, dtype=np.float32), 16000, format="FLAC")
    payload = payload_stream.getvalue()
    row = OutputRow(
        audio=payload,
        audio_sha256=hashlib.sha256(payload).hexdigest(),
        text="hej",
        alignment_text="hej",
        alignment_word_map=("hej",),
        segment_id="a" * 64,
        source_file_id="source",
        source_start_ms=0,
        source_end_ms=10,
        source_duration_ms=10,
        duration_ms=10,
        speaker_ids=(),
        proposal_start_ms=0,
        proposal_end_ms=10,
        alignment_score=1.0,
        alignment_score_type="test",
        start_drift_ms=0,
        end_drift_ms=0,
        vad_speech_ratio=1.0,
        alignment_backend="test",
        pipeline_version="test",
        pipeline_config_sha256="b" * 64,
    )
    path = tmp_path / "valid.parquet"
    pq.write_table(_rows_table([row]), path)
    validate_local_shard(
        path,
        expected_pipeline_version="test",
        expected_pipeline_config_sha256="b" * 64,
        expected_row_count=1,
    )
    broken = tmp_path / "broken.parquet"
    pq.write_table(_rows_table([row.model_copy(update={"duration_ms": 9})]), broken)
    with pytest.raises(VerificationError, match="duration"):
        validate_local_shard(
            broken,
            expected_pipeline_version="test",
            expected_pipeline_config_sha256="b" * 64,
        )
    second_row = _rows_table([row, row])
    duration_column = second_row.column_names.index("duration_ms")
    second_row = second_row.set_column(
        duration_column, "duration_ms", pa.array([10, 9], type=pa.int32())
    )
    multi_row = tmp_path / "multi-row.parquet"
    pq.write_table(second_row, multi_row)
    with pytest.raises(VerificationError, match="duration"):
        validate_local_shard(
            multi_row,
            expected_pipeline_version="test",
            expected_pipeline_config_sha256="b" * 64,
            expected_row_count=2,
        )

    for field, value in (
        ("pipeline_version", "arbitrary-future-version"),
        ("pipeline_config_sha256", "c" * 64),
    ):
        mismatched = tmp_path / f"mismatched-{field}.parquet"
        pq.write_table(_rows_table([row.model_copy(update={field: value})]), mismatched)
        with pytest.raises(VerificationError, match="pipeline"):
            validate_local_shard(
                mismatched,
                expected_pipeline_version="test",
                expected_pipeline_config_sha256="b" * 64,
            )


def test_missing_repository_is_created_private_before_initialisation() -> None:
    """A missing repository is created private and checked before its card commit."""
    hub = MemoryHub(missing=True)
    commit = initialise_private_dataset(hub, "org/p1", card=make_card())
    assert commit == "a" * 40
    assert hub.created
    assert hub.commits == [("README.md", ".gitattributes", "LICENSE")]


def test_post_commit_privacy_failure_stops_before_verification(tmp_path: Path) -> None:
    """A visibility incident stops before path or dataset verification."""
    path = tmp_path / "one.parquet"
    write_valid_shard(path)
    hub = MemoryHub(flip_public=True)
    with pytest.raises(PrivacyError):
        publish_batch(
            hub,
            "org/p1",
            "batch",
            [LocalShard(path, "one.parquet", 1)],
            expected_pipeline_version="test",
            expected_pipeline_config_sha256="b" * 64,
        )
    assert not hub.loaded


def test_public_or_unknown_visibility_aborts_before_commit() -> None:
    """Public, unknown, and malformed visibility never receive a commit."""
    for visibility in (False, None, "private"):
        hub = MemoryHub(private=visibility)
        with pytest.raises(PrivacyError):
            initialise_private_dataset(hub, "org/p1", card=make_card())
        assert not hub.commits


def test_purge_requires_and_follows_durable_verification(tmp_path: Path) -> None:
    """Purging is impossible before, and happens after, durable recording."""
    path = tmp_path / "one.parquet"
    write_valid_shard(path)
    events: list[str] = []
    with pytest.raises(PublicationError):
        publish_batch(
            MemoryHub(),
            "org/p1",
            "batch",
            [LocalShard(path, "one.parquet", 1)],
            expected_pipeline_version="test",
            expected_pipeline_config_sha256="b" * 64,
            purge_callback=lambda paths: events.append("purge"),
        )
    assert path.exists()

    def purge(paths: tuple[Path, ...]) -> None:
        """Remove every artefact supplied by the publisher."""
        events.append("purge")
        for candidate in paths:
            candidate.unlink()

    evidence = publish_batch(
        MemoryHub(),
        "org/p1",
        "batch",
        [LocalShard(path, "one.parquet", 1)],
        expected_pipeline_version="test",
        expected_pipeline_config_sha256="b" * 64,
        durable_verification=lambda _: events.append("durable"),
        purge_callback=purge,
    )
    assert evidence.state.value == "purged"
    assert events[-2:] == ["durable", "purge"]
    assert not path.exists()


def test_racing_competitor_cannot_overwrite_or_lose_ledger_batch(
    tmp_path: Path,
) -> None:
    """A stale parent aborts publication and leaves a batch resumable."""
    path = tmp_path / "one.parquet"
    write_valid_shard(path)
    competitor_bytes = b"competitor owns this path"
    hub = RacingHub("one.parquet", competitor_bytes)
    database = tmp_path / "ledger.sqlite"

    with Ledger(database) as ledger:
        ledger.register_batch("batch", pipeline_digest="a" * 64)
        ledger.register_shard(
            "shard",
            path="one.parquet",
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            byte_size=path.stat().st_size,
            row_count=1,
            batch_id="batch",
        )
        ledger.transition_batch("batch", LedgerState.PROCESSING)
        ledger.transition_batch("batch", LedgerState.SHARDED)

        with pytest.raises(RuntimeError, match="stale parent"):
            publish_batch(
                hub,
                "org/p1",
                "batch",
                [LocalShard(path, "one.parquet", 1)],
                expected_pipeline_version="test",
                expected_pipeline_config_sha256="b" * 64,
                ledger=ledger,
            )

        assert hub.revisions == ["b" * 40]
        assert hub.parents == ["b" * 40]
        assert hub.files["one.parquet"] == competitor_bytes
        assert ledger.batch("batch").state is LedgerState.SHARDED
        assert ledger.batch("batch").commit_id is None
        with pytest.raises(AllowListError, match="collision"):
            publish_batch(
                hub,
                "org/p1",
                "batch",
                [LocalShard(path, "one.parquet", 1)],
                expected_pipeline_version="test",
                expected_pipeline_config_sha256="b" * 64,
                ledger=ledger,
            )
        assert ledger.batch("batch").state is LedgerState.SHARDED
        assert hub.files["one.parquet"] == competitor_bytes


class RacingHub(MemoryHub):
    """Hub fake that races a competing commit after an immutable listing."""

    def __init__(self, race_path: str, race_bytes: bytes) -> None:
        """Set up a fake with a colliding competitor path."""
        super().__init__()
        self.head = "b" * 40
        self.race_path = race_path
        self.race_bytes = race_bytes
        self.race = True
        self.parents: list[str | None] = []
        self.revisions: list[str | None] = []

    def create_commit(
        self,
        repo_id: str,
        operations: c.Iterable[UploadOperation],
        *,
        repo_type: str,
        commit_message: str,
        parent_commit: str | None = None,
    ) -> object:
        """Reject stale parents instead of applying their operations.

        Returns:
            Fake commit metadata.

        Raises:
            RuntimeError:
                If the requested parent is no longer the current head.
        """
        self.parents.append(parent_commit)
        if parent_commit != self.head:
            raise RuntimeError("stale parent")
        result = super().create_commit(
            repo_id,
            operations,
            repo_type=repo_type,
            commit_message=commit_message,
            parent_commit=parent_commit,
        )
        self.head = self.commit_id
        return result

    def get_paths_info(
        self, repo_id: str, paths: list[str], *, repo_type: str, revision: str
    ) -> list[object]:
        """Return the old immutable snapshot before racing a competitor."""
        self.revisions.append(revision)
        snapshot = super().get_paths_info(
            repo_id, paths, repo_type=repo_type, revision=revision
        )
        if self.race:
            self.race = False
            self.files[self.race_path] = self.race_bytes
            self.head = "c" * 40
        return snapshot

    def repo_info(
        self, repo_id: str, *, repo_type: str, revision: str | None = None
    ) -> object:
        """Expose the current branch head to the publisher.

        Returns:
            Fake repository metadata.
        """
        self.privacy_checks += 1
        return SimpleNamespace(private=self.private, sha=self.head)


def test_remote_digest_stream_does_not_retain_shard_bytes() -> None:
    """Remote digesting consumes chunks without joining a shard payload."""
    hub = MemoryHub()
    hub.files["one.parquet"] = b"x" * 1024
    evidence = ShardEvidence(
        path="one.parquet",
        byte_size=1024,
        row_count=1,
        sha256=hashlib.sha256(b"x" * 1024).hexdigest(),
    )
    digest, retained = _stream_remote(
        hub, "org/p1", item=evidence, revision="a" * 40, retain=False
    )
    assert digest == evidence.sha256
    assert retained is None


def test_remote_path_collision_is_refused(tmp_path: Path) -> None:
    """Publishing never overwrites a path owned by an existing publication."""
    path = tmp_path / "one.parquet"
    write_valid_shard(path)
    hub = MemoryHub(sha="b" * 40, existing_paths=("one.parquet",))
    with pytest.raises(AllowListError, match="collision"):
        publish_batch(
            hub,
            "org/p1",
            "batch",
            [LocalShard(path, "one.parquet", 1)],
            expected_pipeline_version="test",
            expected_pipeline_config_sha256="b" * 64,
        )
    assert not hub.commits


def test_stream_decode_failure_retains_the_pending_batch(tmp_path: Path) -> None:
    """An empty streaming shard cannot trigger local purging."""
    path = tmp_path / "one.parquet"
    write_valid_shard(path)
    hub = MemoryHub(decode_empty=True)
    purged = False

    def purge(paths: tuple[Path, ...]) -> None:
        """Record an unexpected purge attempt."""
        nonlocal purged
        purged = True

    with pytest.raises(VerificationError):
        publish_batch(
            hub,
            "org/p1",
            "batch",
            [LocalShard(path, "one.parquet", 1)],
            expected_pipeline_version="test",
            expected_pipeline_config_sha256="b" * 64,
            durable_verification=lambda evidence: None,
            purge_callback=purge,
        )
    assert path.exists()
    assert not purged


def test_symlinks_and_unexpected_staging_entries_are_rejected(tmp_path: Path) -> None:
    """Symlinks and files outside the explicit staging allow-list are rejected."""
    real = tmp_path / "real.parquet"
    real.write_bytes(b"data")
    link = tmp_path / "link.parquet"
    link.symlink_to(real)
    with pytest.raises(AllowListError):
        publish_batch(
            MemoryHub(),
            "org/p1",
            "batch",
            [LocalShard(link, "link.parquet", 1)],
            expected_pipeline_version="test",
            expected_pipeline_config_sha256="b" * 64,
        )

    unexpected = tmp_path / "unexpected.txt"
    unexpected.write_text("no")
    with pytest.raises(AllowListError):
        validate_staging_directory(tmp_path, [Path("real.parquet")])
