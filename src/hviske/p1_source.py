"""Bounded, authenticated access to the pinned P1 source datasets.

The source repositories contain very large embedded audio and transcript objects.  This
module keeps planning and discovery metadata-only, stores transcript lookups as SQLite
pointers, and reads the payload for one selected programme at a time.
"""

from __future__ import annotations

import collections.abc as c
import contextlib
import dataclasses
import decimal
import io
import json
import logging
import math
import numbers
import re
import sqlite3
import typing as t
import unicodedata
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from .p1_contracts import SourceWord, annotate_source_words

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_INDEX_PROGRESS_INTERVAL = 10_000
_DEFAULT_MAX_DECODED_AUDIO_BYTES = 2 * 1024**3


class _SignedUrlFilter(logging.Filter):
    """Remove credentials from accidental dependency log records."""

    _url_pattern = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)

    def filter(self, record: logging.LogRecord) -> bool:
        """Redact query strings and signed URL parameters in-place.

        Returns:
            Always ``True`` so the record remains eligible for other handlers.
        """
        message = record.getMessage()
        record.msg = self._url_pattern.sub("<url-redacted>", message)
        record.args = ()
        if record.exc_text:
            record.exc_text = self._url_pattern.sub("<url-redacted>", record.exc_text)
        return True


def _validate_database_path(path: Path) -> None:
    """Reject database paths that could escape the owned scratch tree.

    Raises:
        ValueError:
            If the database or one of its sidecars is an unexpected path.
    """
    if path.is_symlink():
        raise ValueError(f"database path must not be a symlink: {path}")
    if path.exists() and not path.is_file():
        raise ValueError(f"database path must be a regular file: {path}")
    if path.parent.is_symlink():
        raise ValueError(f"database parent must not be a symlink: {path.parent}")
    for suffix in ("-wal", "-shm"):
        sidecar = Path(f"{path}{suffix}")
        if sidecar.is_symlink():
            raise ValueError(f"database sidecar must not be a symlink: {sidecar}")
        if sidecar.exists() and not sidecar.is_file():
            raise ValueError(f"database sidecar must be a regular file: {sidecar}")


_SIGNED_URL_FILTER = _SignedUrlFilter()
_TRANSPORT_LOGGER_NAMES = ("httpx", "httpcore", "huggingface_hub", "fsspec")


def harden_p1_logging() -> None:
    """Harden dependency logging and redact URLs on all active handlers.

    Hydra installs its handlers before invoking the pipeline, so this function is
    intended to run at the beginning of :func:`run_pipeline`.  It is safe to call
    repeatedly, including after a handler has been added, and leaves the P1
    application loggers at their configured levels.
    """
    for name in _TRANSPORT_LOGGER_NAMES:
        logging.getLogger(name).setLevel(logging.WARNING)

    handlers: list[logging.Handler] = list(logging.root.handlers)
    if logging.lastResort is not None:
        handlers.append(logging.lastResort)
    for value in logging.Logger.manager.loggerDict.values():
        if isinstance(value, logging.Logger):
            handlers.extend(value.handlers)

    seen: set[int] = set()
    for handler in handlers:
        if id(handler) in seen:
            continue
        seen.add(id(handler))
        if not any(isinstance(item, _SignedUrlFilter) for item in handler.filters):
            handler.addFilter(_SIGNED_URL_FILTER)


logger.addFilter(_SIGNED_URL_FILTER)
_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


@dataclasses.dataclass(frozen=True)
class ParsedTranscript:
    """One transcript with timed words and lossless source-text ownership evidence."""

    file_id: str
    text: str
    words: tuple[SourceWord, ...]
    metadata: tuple[tuple[str, str], ...] = ()
    zero_duration_tokens_omitted: int = 0
    untimed_tokens_owned: int = 0
    ambiguous_source_text_records: int = 0


class SourceError(RuntimeError):
    """Base class for source access failures."""


class InvalidSourceRecord(SourceError, ValueError):
    """Raised when a source row cannot be interpreted safely."""


class DecodedAudioTooLarge(InvalidSourceRecord):
    """Raised before decoding PCM that exceeds the configured hard limit."""


class InvalidSourceTimestamp(InvalidSourceRecord):
    """Raised when source word timestamps cannot form a valid timeline."""


class SourceObjectTooLarge(SourceError, ValueError):
    """Raised before opening a source object larger than the configured limit."""


class SourceSelectionError(SourceError, ValueError):
    """Raised when a requested source programme has no valid joined pointer."""


@dataclasses.dataclass(frozen=True)
class SourceShard:
    """Metadata for one immutable audio Parquet object."""

    path: str
    byte_size: int
    revision: str = ""
    oid: str | None = None


@dataclasses.dataclass(frozen=True)
class TranscriptIndexRejection:
    """Metadata-only evidence for a row omitted from the pointer index."""

    reason: str
    file_id: str | None
    path: str
    row_group: int
    row_index: int


class TranscriptOverAudio(InvalidSourceTimestamp):
    """Raised when a valid transcript endpoint exceeds decoded source audio."""


@dataclasses.dataclass(frozen=True)
class _PendingLexicalRecord:
    """Lexical source text waiting for its bounded ownership lookahead."""

    speaker_id: str | None
    point_ms: int | None


class _RemoteRowPointerMixin:
    """Provide one implementation for immutable remote row coordinates."""

    @property
    def remote_row_pointer(self) -> tuple[str, int, int]:
        """Immutable shard, row-group, and row offset."""
        return self._remote_row_path(), self.row_group, self.row_index

    def _remote_row_path(self) -> str:
        """Return the object path represented by this pointer."""
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class AudioPointer(_RemoteRowPointerMixin):
    """A source-audio row location discovered without reading the audio column."""

    file_id: str
    shard: SourceShard
    row_group: int
    row_index: int
    metadata: tuple[tuple[str, str], ...] = ()

    def _remote_row_path(self) -> str:
        """Return the audio shard path represented by this pointer."""
        return self.shard.path


@dataclasses.dataclass(frozen=True)
class TranscriptPointer(_RemoteRowPointerMixin):
    """A remote row location; no transcript content is retained."""

    file_id: str
    path: str
    row_group: int
    row_index: int
    revision: str
    byte_size: int
    metadata: tuple[tuple[str, str], ...] = ()

    def _remote_row_path(self) -> str:
        """Return the transcript object path represented by this pointer."""
        return self.path


@dataclasses.dataclass(frozen=True)
class _TimedOwnershipAnchor:
    """One positive-duration speaker and timestamp ownership anchor."""

    speaker_id: str | None
    start_ms: int
    end_ms: int


TranscriptWord = SourceWord

# This value is carried only between the metadata sampler and pointer rebuild.  It
# must remain small enough to exclude accidental source payloads from the reservoir.
_AUDIO_POINTER_METADATA_KEY = "_p1_audio_pointer_metadata"
_AUDIO_POINTER_METADATA_MAX_BYTES = 64 * 1024


@dataclasses.dataclass(frozen=True)
class SourcePlan:
    """Hub metadata needed to plan a bounded run."""

    audio_repository: str
    transcript_repository: str
    audio_revision: str
    transcript_revision: str
    audio_shards: tuple[SourceShard, ...]
    transcript_objects: tuple[tuple[str, int, str | None], ...]


@dataclasses.dataclass(frozen=True)
class ParsedAudio:
    """One fetched audio row, retaining its source shape and sampling rate."""

    file_id: str
    value: bytes | np.ndarray | Path
    sampling_rate: int
    channels: int = 1

    @property
    def duration_ms(self) -> int | None:
        """Nearest-millisecond decoded duration when available."""
        frames = self.frame_count
        if frames is None or self.sampling_rate <= 0:
            return None
        return (frames * 1000 + self.sampling_rate // 2) // self.sampling_rate

    @property
    def frame_count(self) -> int | None:
        """Decoded frames when this row contains an in-memory array."""
        if not isinstance(self.value, np.ndarray):
            return None
        if self.value.ndim == 1:
            return int(self.value.shape[0])
        if self.value.ndim == 2:
            return int(self.value.shape[0])
        return None


class TranscriptPointerIndex:
    """Disk-backed file-id to transcript-row index.

    Only identifiers, remote row coordinates, and scalar stratification metadata are
    written.  In particular, this database has no columns capable of holding text,
    words, or audio.
    """

    def __init__(
        self, path: Path, scratch_guard: c.Callable[[], None] | None = None
    ) -> None:
        """Create or open an index at ``path``.

        Args:
            path:
                Database path in the owned scratch tree.
            scratch_guard (optional):
                Callback that rejects a run after scratch growth.
        """
        self.path = Path(path)
        self._scratch_guard = scratch_guard
        _validate_database_path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialise_database()

    def _initialise_database(self) -> None:
        """Create the index schema and account for its durable scratch state."""
        with contextlib.closing(self._connect()) as connection, connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS transcript_pointers (
                    file_id TEXT PRIMARY KEY NOT NULL,
                    path TEXT NOT NULL,
                    row_group INTEGER NOT NULL,
                    row_index INTEGER NOT NULL,
                    revision TEXT NOT NULL,
                    byte_size INTEGER NOT NULL,
                    metadata TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS rejections (
                    reason TEXT NOT NULL,
                    file_id TEXT,
                    path TEXT NOT NULL,
                    row_group INTEGER NOT NULL,
                    row_index INTEGER NOT NULL
                );
                """
            )
        self._guard_scratch()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _guard_scratch(self) -> None:
        """Check the scratch quota after a database transaction is durable."""
        if self._scratch_guard is not None:
            self._scratch_guard()

    def __len__(self) -> int:
        """Return the number of indexed transcripts."""
        with contextlib.closing(self._connect()) as connection, connection:
            return int(
                connection.execute(
                    "SELECT count(*) FROM transcript_pointers"
                ).fetchone()[0]
            )

    def add(self, pointer: TranscriptPointer) -> None:
        """Add one pointer, rejecting duplicate file identifiers.

        Raises:
            InvalidSourceRecord:
                If the file identifier has already been indexed.
        """
        with contextlib.closing(self._connect()) as connection, connection:
            try:
                connection.execute(
                    """INSERT INTO transcript_pointers
                    (file_id, path, row_group, row_index, revision, byte_size, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        pointer.file_id,
                        pointer.path,
                        pointer.row_group,
                        pointer.row_index,
                        pointer.revision,
                        pointer.byte_size,
                        json.dumps(pointer.metadata, ensure_ascii=False),
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise InvalidSourceRecord(
                    f"duplicate transcript file_id: {pointer.file_id}"
                ) from error
        self._guard_scratch()

    def clear(self) -> None:
        """Recreate this index without retaining SQLite high-water space."""
        _validate_database_path(self.path)
        self._remove_database_files(self.path)
        self._initialise_database()

    @staticmethod
    def _remove_database_files(path: Path) -> None:
        """Unlink only the validated database files owned by this index."""
        for database_file in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
            database_file.unlink(missing_ok=True)

    def get(self, file_id: str) -> TranscriptPointer | None:
        """Return a pointer without reading transcript content."""
        with contextlib.closing(self._connect()) as connection, connection:
            row = connection.execute(
                (
                    "SELECT file_id, path, row_group, row_index, revision, byte_size, "
                    "metadata FROM transcript_pointers WHERE file_id = ?"
                ),
                (file_id,),
            ).fetchone()
        if row is None:
            return None
        decoded_metadata = json.loads(row[6])
        if isinstance(decoded_metadata, dict):
            metadata = tuple(sorted(decoded_metadata.items()))
        else:
            metadata = tuple(
                (item[0], item[1])
                for item in decoded_metadata
                if isinstance(item, list) and len(item) == 2
            )
        return TranscriptPointer(
            file_id=row[0],
            path=row[1],
            row_group=row[2],
            row_index=row[3],
            revision=row[4],
            byte_size=row[5],
            metadata=metadata,
        )

    @classmethod
    def rebuild(
        cls, path: Path, scratch_guard: c.Callable[[], None] | None = None
    ) -> t.Self:
        """Recreate an empty index without retaining SQLite high-water space.

        Args:
            path:
                Database path in the owned scratch tree.
            scratch_guard (optional):
                Callback that rejects a run after scratch growth.

        Returns:
            A fresh disk-backed pointer index.
        """
        database_path = Path(path)
        _validate_database_path(database_path)
        cls._remove_database_files(database_path)
        return cls(database_path, scratch_guard=scratch_guard)

    def reject(self, rejection: TranscriptIndexRejection) -> None:
        """Record metadata-only rejection evidence."""
        with contextlib.closing(self._connect()) as connection, connection:
            connection.execute(
                "INSERT INTO rejections VALUES (?, ?, ?, ?, ?)",
                dataclasses.astuple(rejection),
            )
        self._guard_scratch()

    def rejections(self) -> tuple[TranscriptIndexRejection, ...]:
        """Return metadata-only index rejection evidence."""
        with contextlib.closing(self._connect()) as connection, connection:
            rows = connection.execute(
                "SELECT reason, file_id, path, row_group, row_index FROM rejections "
                "ORDER BY path, row_group, row_index"
            ).fetchall()
        return tuple(TranscriptIndexRejection(*row) for row in rows)


class HfP1Source:
    """Authenticated source adapter for ``syvai/p1`` and its transcripts."""

    def __init__(
        self,
        audio_repository: str = "syvai/p1",
        transcript_repository: str = "syvai/p1-transcripts",
        *,
        token: str | bool | None = True,
        local_root: Path | None = None,
        max_source_object_bytes: int = 6_197_291_423,
        max_batch_rows: int = 1024,
        max_batch_bytes: int = 64 * 1024 * 1024,
        max_decoded_audio_bytes: int = _DEFAULT_MAX_DECODED_AUDIO_BYTES,
    ) -> None:
        """Create an adapter without downloading source data.

        Args:
            token (optional): Hugging Face token passed to authenticated clients.
            audio_repository (optional): Pinned audio repository identifier.
            transcript_repository (optional): Pinned transcript repository identifier.
            local_root (optional): Test or mirror root containing both repositories.
            max_source_object_bytes (optional): Hard object-size limit before retrieval.
            max_batch_rows (optional): Maximum projected rows held in one Arrow batch.
            max_batch_bytes (optional): Maximum projected Arrow batch size.
            max_decoded_audio_bytes (optional): Hard limit for decoded float32 PCM.
                The default is 2 GiB, which covers the observed roughly 58-minute,
                48 kHz mono P1 programme while bounding decompression expansion.

        Raises:
            ValueError:
                If a configured bound is not positive.
        """
        if max_source_object_bytes <= 0:
            raise ValueError("max_source_object_bytes must be positive")
        if max_batch_rows <= 0 or max_batch_bytes <= 0:
            raise ValueError("batch bounds must be positive")
        if max_decoded_audio_bytes <= 0:
            raise ValueError("max_decoded_audio_bytes must be positive")
        self.token = token
        self.audio_repository = audio_repository
        self.transcript_repository = transcript_repository
        self.local_root = Path(local_root) if local_root is not None else None
        self.max_source_object_bytes = max_source_object_bytes
        self.max_batch_rows = max_batch_rows
        self.max_batch_bytes = max_batch_bytes
        self.max_decoded_audio_bytes = max_decoded_audio_bytes
        self._api: object | None = None
        self._fs: object | None = None

    def build_transcript_index(
        self,
        *,
        revision: str,
        path: Path,
        objects: c.Iterable[tuple[str, int, str | None]],
        source_file_id: str | None = None,
        scratch_guard: c.Callable[[], None] | None = None,
    ) -> TranscriptPointerIndex:
        """Build a bounded SQLite pointer index from transcript metadata columns.

        Args:
            revision:
                Immutable transcript revision.
            path:
                Database path in the owned scratch tree.
            objects:
                Transcript objects to scan using metadata columns only.
            source_file_id (optional):
                If set, stop after indexing this file identifier.
            scratch_guard (optional):
                Callback that rejects the run after committed index growth.

        Returns:
            The disk-backed pointer index.

        Raises:
            SourceObjectTooLarge:
                If a transcript object exceeds the configured limit.
            SourceSelectionError:
                If ``source_file_id`` is requested but no valid pointer is found.
        """
        index = TranscriptPointerIndex.rebuild(path, scratch_guard=scratch_guard)
        indexed_count = 0
        for shard_path, byte_size, _oid in sorted(objects, key=lambda item: item[0]):
            if byte_size > self.max_source_object_bytes:
                raise SourceObjectTooLarge(
                    f"transcript object {shard_path} exceeds the configured limit"
                )
            with self._parquet(
                shard_path, repository=self.transcript_repository, revision=revision
            ) as parquet:
                columns = _transcript_metadata_columns(parquet.schema_arrow.names)
                for row_group in range(parquet.num_row_groups):
                    row_offset = 0
                    for batch in parquet.iter_batches(
                        row_groups=[row_group],
                        columns=columns,
                        batch_size=self.max_batch_rows,
                    ):
                        _check_batch_bytes(batch, self.max_batch_bytes)
                        for row in batch.to_pylist():
                            file_id = _file_id(row)
                            if file_id is None:
                                index.reject(
                                    TranscriptIndexRejection(
                                        "missing_file_id",
                                        None,
                                        shard_path,
                                        row_group,
                                        row_offset,
                                    )
                                )
                            else:
                                if (
                                    source_file_id is not None
                                    and file_id != source_file_id
                                ):
                                    row_offset += 1
                                    continue
                                pointer = TranscriptPointer(
                                    file_id=file_id,
                                    path=shard_path,
                                    row_group=row_group,
                                    row_index=row_offset,
                                    revision=revision,
                                    byte_size=byte_size,
                                    metadata=_scalar_metadata(row),
                                )
                                try:
                                    index.add(pointer)
                                    indexed_count += 1
                                    if indexed_count % _INDEX_PROGRESS_INTERVAL == 0:
                                        logger.info(
                                            "Transcript index progress: %d pointers",
                                            indexed_count,
                                        )
                                    if source_file_id is not None:
                                        logger.info(
                                            "Transcript index complete: %d pointers "
                                            "(targeted)",
                                            indexed_count,
                                        )
                                        return index
                                except InvalidSourceRecord:
                                    index.reject(
                                        TranscriptIndexRejection(
                                            "duplicate_file_id",
                                            file_id,
                                            shard_path,
                                            row_group,
                                            row_offset,
                                        )
                                    )
                            row_offset += 1
        logger.info("Transcript index complete: %d pointers", indexed_count)
        if source_file_id is not None:
            raise SourceSelectionError(
                "requested source_file_id has no valid transcript pointer"
            )
        return index

    @contextlib.contextmanager
    def _parquet(
        self, path: str, *, repository: str, revision: str
    ) -> c.Iterator[pq.ParquetFile]:
        if self.local_root is not None:
            local = self._local_path(repository, path)
            yield pq.ParquetFile(local)
            return
        filesystem = self._filesystem()
        handle = filesystem.open(
            f"datasets/{repository}/{path}", "rb", revision=revision
        )
        try:
            yield pq.ParquetFile(handle)
        finally:
            handle.close()

    def _filesystem(self) -> object:
        if self._fs is None:
            from huggingface_hub import HfFileSystem

            self._fs = HfFileSystem(token=self.token)
        return self._fs

    def _local_path(self, repository: str, path: str) -> Path:
        assert self.local_root is not None
        candidates = (
            self.local_root / repository / path,
            self.local_root / repository.replace("/", "_") / path,
            self.local_root / path,
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(path)

    def check_revisions(self, *, audio_revision: str, transcript_revision: str) -> bool:
        """Resolve both immutable dataset revisions without downloading objects.

        Returns:
            ``True`` when both authenticated repository lookups succeed.
        """
        _validate_revision(audio_revision)
        _validate_revision(transcript_revision)
        api = self._client()
        api.repo_info(
            self.audio_repository, repo_type="dataset", revision=audio_revision
        )
        api.repo_info(
            self.transcript_repository,
            repo_type="dataset",
            revision=transcript_revision,
        )
        return True

    def _client(self) -> object:
        if self._api is None:
            from huggingface_hub import HfApi

            self._api = HfApi(token=self.token)
        return self._api

    def fetch_transcript(self, pointer: TranscriptPointer) -> ParsedTranscript:
        """Fetch and parse exactly one transcript row addressed by ``pointer``.

        Returns:
            The parsed transcript and its validated words.
        """
        self._check_size(pointer.path, pointer.byte_size)
        with self._parquet(
            pointer.path,
            repository=self.transcript_repository,
            revision=pointer.revision,
        ) as parquet:
            columns = _transcript_payload_columns(parquet.schema_arrow.names)
            row = _read_one_row(
                parquet=parquet,
                row_group=pointer.row_group,
                row_index=pointer.row_index,
                columns=columns,
                max_batch_bytes=self.max_batch_bytes,
            )
        parsed = parse_transcript_row(row=row, expected_file_id=pointer.file_id)
        return dataclasses.replace(parsed, metadata=pointer.metadata)

    def _check_size(self, path: str, byte_size: int) -> None:
        if byte_size < 0:
            raise InvalidSourceRecord(f"source object {path} has a negative size")
        if byte_size > self.max_source_object_bytes:
            raise SourceObjectTooLarge(
                f"source object {path} ({byte_size} bytes) exceeds "
                f"{self.max_source_object_bytes} bytes"
            )

    def iter_programmes(
        self, *, shard: SourceShard, index: object | None = None
    ) -> c.Iterator[dict[str, object]]:
        """Compatibility alias for metadata-only programme discovery.

        ``index`` is accepted for source-adapter compatibility but is intentionally
        unused: joining must happen after discovery and never require audio decoding.

        Yields:
            Metadata rows with the embedded audio column omitted.
        """
        del index
        yield from self.iter_programme_metadata(shard=shard)

    def iter_programme_metadata(
        self, *, shard: SourceShard, batch_size: int = 1024
    ) -> c.Iterator[dict[str, object]]:
        """Yield source metadata while projecting out the embedded audio column."""
        self._check_size(shard.path, shard.byte_size)
        with self._parquet(
            shard.path, repository=self.audio_repository, revision=shard.revision
        ) as parquet:
            columns = tuple(
                name for name in parquet.schema_arrow.names if name.lower() != "audio"
            )
            effective_batch_size = self._batch_size(batch_size)
            for row_group in range(parquet.num_row_groups):
                row_index = 0
                for batch in parquet.iter_batches(
                    row_groups=[row_group],
                    columns=columns,
                    batch_size=effective_batch_size,
                ):
                    _check_batch_bytes(batch, self.max_batch_bytes)
                    for raw_row in batch.to_pylist():
                        row = dict(raw_row)
                        row["source_row_group"] = row_group
                        row["source_row_index"] = row_index
                        yield row
                        row_index += 1

    def _batch_size(self, requested: int) -> int:
        if requested <= 0:
            raise ValueError("batch_size must be positive")
        return min(requested, self.max_batch_rows)

    def list_audio_shards(self, *, revision: str) -> tuple[dict[str, object], ...]:
        """List audio Parquet objects from authenticated tree metadata only.

        Returns:
            Deterministically ordered path, size, and object-id metadata.
        """
        _validate_revision(revision)
        if self.local_root is not None:
            entries = self._local_entries(self.audio_repository)
        else:
            api = self._client()
            entries = tuple(
                self._tree_entry(item)
                for item in api.list_repo_tree(
                    self.audio_repository,
                    recursive=True,
                    revision=revision,
                    repo_type="dataset",
                )
            )
        return tuple(
            {"path": path, "size": size, "oid": oid}
            for path, size, oid in sorted(entries, key=lambda entry: entry[0])
            if path.lower().endswith(".parquet")
        )

    def _local_entries(
        self, repository: str
    ) -> tuple[tuple[str, int, str | None], ...]:
        assert self.local_root is not None
        root = self.local_root / repository
        if not root.exists():
            root = self.local_root / repository.replace("/", "_")
        return tuple(
            (str(path.relative_to(root)), path.stat().st_size, None)
            for path in root.rglob("*")
            if path.is_file()
        )

    @staticmethod
    def _tree_entry(item: object) -> tuple[str, int, str | None]:
        if isinstance(item, dict):
            return (
                str(item.get("path", "")),
                int(item.get("size") or 0),
                item.get("oid"),
            )
        return (
            str(getattr(item, "path", "")),
            int(getattr(item, "size", 0) or 0),
            t.cast(str | None, getattr(item, "oid", None)),
        )

    def plan(self, *, audio_revision: str, transcript_revision: str) -> SourcePlan:
        """Plan from Hub repository and tree metadata only.

        This method never constructs a Datasets object and never iterates a Parquet,
        audio, or transcript row.

        Returns:
            Repository and object metadata for the bounded run.
        """
        _validate_revision(audio_revision)
        _validate_revision(transcript_revision)
        if self.local_root is not None:
            audio_entries = self._local_entries(self.audio_repository)
            transcript_entries = self._local_entries(self.transcript_repository)
        else:
            api = self._client()
            api.repo_info(
                self.audio_repository, repo_type="dataset", revision=audio_revision
            )
            api.repo_info(
                self.transcript_repository,
                repo_type="dataset",
                revision=transcript_revision,
            )
            audio_entries = tuple(
                self._tree_entry(item)
                for item in api.list_repo_tree(
                    self.audio_repository,
                    recursive=True,
                    revision=audio_revision,
                    repo_type="dataset",
                )
            )
            transcript_entries = tuple(
                self._tree_entry(item)
                for item in api.list_repo_tree(
                    self.transcript_repository,
                    recursive=True,
                    revision=transcript_revision,
                    repo_type="dataset",
                )
            )
        shards = tuple(
            SourceShard(path=path, byte_size=size, revision=audio_revision, oid=oid)
            for path, size, oid in audio_entries
            if path.lower().endswith(".parquet")
        )
        shards = tuple(sorted(shards, key=lambda item: item.path))
        transcripts = tuple(
            sorted(
                (
                    entry
                    for entry in transcript_entries
                    if entry[0].lower().endswith(".parquet")
                ),
                key=lambda entry: entry[0],
            )
        )
        self._check_sizes(shards)
        return SourcePlan(
            audio_repository=self.audio_repository,
            transcript_repository=self.transcript_repository,
            audio_revision=audio_revision,
            transcript_revision=transcript_revision,
            audio_shards=shards,
            transcript_objects=transcripts,
        )

    def _check_sizes(self, shards: c.Iterable[SourceShard]) -> None:
        for shard in shards:
            self._check_size(shard.path, shard.byte_size)

    def retrieve_audio(self, *, programme: object, shard: SourceShard) -> object:
        """Retrieve one programme's audio payload by locating its row pointer.

        Returns:
            The raw audio payload; :meth:`fetch_audio` additionally exposes its rate.

        Raises:
            FileNotFoundError:
                If the programme identifier is not present in ``shard``.
            InvalidSourceRecord:
                If the programme has no valid identifier.
        """
        file_id = (
            programme if isinstance(programme, str) else _programme_file_id(programme)
        )
        if file_id is None:
            raise InvalidSourceRecord("programme has no file_id")
        pointer = next(
            (
                item
                for item in self.iter_programme_pointers(shard=shard)
                if item.file_id == file_id
            ),
            None,
        )
        if pointer is None:
            raise FileNotFoundError(f"source file_id not found in {shard.path}")
        return self.fetch_audio(pointer=pointer).value

    def fetch_audio(
        self,
        *,
        pointer: AudioPointer | None = None,
        shard: SourceShard | None = None,
        row_group: int | None = None,
        row_index: int | None = None,
    ) -> ParsedAudio:
        """Fetch and parse one audio row after enforcing its object-size limit.

        Returns:
            The audio payload and its source sampling rate.

        Raises:
            ValueError:
                If row coordinates are incomplete.
        """
        expected_file_id: str | None = None
        if pointer is not None:
            shard = pointer.shard
            row_group = pointer.row_group
            row_index = pointer.row_index
            expected_file_id = pointer.file_id
        if shard is None or row_group is None or row_index is None:
            raise ValueError(
                "an audio pointer or complete row coordinates are required"
            )
        self._check_size(shard.path, shard.byte_size)
        with self._parquet(
            shard.path, repository=self.audio_repository, revision=shard.revision
        ) as parquet:
            columns = tuple(
                name
                for name in parquet.schema_arrow.names
                if name.lower() in {"audio", "file_id"}
            )
            row = _read_one_row(
                parquet=parquet,
                row_group=row_group,
                row_index=row_index,
                columns=columns,
                max_batch_bytes=self.max_batch_bytes,
            )
            sampling_rate = _audio_sampling_rate(parquet)
            channels = _audio_channels(parquet)
        return parse_audio_row(
            row,
            expected_file_id=expected_file_id,
            default_sampling_rate=sampling_rate,
            default_channels=channels,
            max_decoded_audio_bytes=self.max_decoded_audio_bytes,
        )

    def iter_programme_pointers(
        self, *, shard: SourceShard, batch_size: int = 1024
    ) -> c.Iterator[AudioPointer]:
        """Yield audio row pointers using metadata columns only."""
        self._check_size(shard.path, shard.byte_size)
        with self._parquet(
            shard.path, repository=self.audio_repository, revision=shard.revision
        ) as parquet:
            columns = tuple(
                name for name in parquet.schema_arrow.names if name.lower() != "audio"
            )
            effective_batch_size = self._batch_size(batch_size)
            for row_group in range(parquet.num_row_groups):
                row_index = 0
                for batch in parquet.iter_batches(
                    row_groups=[row_group],
                    columns=columns,
                    batch_size=effective_batch_size,
                ):
                    _check_batch_bytes(batch, self.max_batch_bytes)
                    for row in batch.to_pylist():
                        file_id = _file_id(row)
                        if file_id is not None:
                            yield AudioPointer(
                                file_id=file_id,
                                shard=shard,
                                row_group=row_group,
                                row_index=row_index,
                                metadata=_scalar_metadata(row),
                            )
                        row_index += 1


P1Source = HfP1Source


def _audio_channels(parquet: pq.ParquetFile) -> int | None:
    """Read an Audio feature's channel count from Parquet schema metadata.

    Returns:
        The declared channel count, or ``None`` when it is absent.
    """
    metadata = parquet.schema_arrow.metadata or {}
    feature_json = metadata.get(b"huggingface")
    if feature_json is None:
        return None
    try:
        value: object = json.loads(feature_json)
    except (TypeError, ValueError):
        return None
    return _find_channels(value)


def _find_channels(value: object) -> int | None:
    if isinstance(value, c.Mapping):
        channels = value.get("channels")
        if isinstance(channels, numbers.Integral) and not isinstance(channels, bool):
            return int(channels)
        for child in value.values():
            found = _find_channels(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_channels(child)
            if found is not None:
                return found
    return None


def _audio_sampling_rate(parquet: pq.ParquetFile) -> int | None:
    """Read an Audio feature's sampling rate from Parquet schema metadata.

    Returns:
        The declared sampling rate, or ``None`` when no feature metadata exists.
    """
    metadata = parquet.schema_arrow.metadata or {}
    feature_json = metadata.get(b"huggingface")
    if feature_json is None:
        return None
    try:
        value: object = json.loads(feature_json)
    except (TypeError, ValueError):
        return None
    return _find_sampling_rate(value)


def _find_sampling_rate(value: object) -> int | None:
    if isinstance(value, c.Mapping):
        rate = value.get("sampling_rate")
        if isinstance(rate, numbers.Integral) and not isinstance(rate, bool):
            return int(rate)
        for child in value.values():
            found = _find_sampling_rate(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_sampling_rate(child)
            if found is not None:
                return found
    return None


def _programme_file_id(programme: object) -> str | None:
    if isinstance(programme, c.Mapping):
        return _file_id(programme)
    value = getattr(programme, "file_id", None)
    return value if isinstance(value, str) and value else None


def _file_id(row: c.Mapping[str, object]) -> str | None:
    value = row.get("file_id", row.get("audio_id"))
    return value if isinstance(value, str) and value else None


def _read_one_row(
    *,
    parquet: pq.ParquetFile,
    row_group: int,
    row_index: int,
    columns: c.Sequence[str],
    max_batch_bytes: int,
) -> dict[str, object]:
    """Read one projected row without materialising its row group.

    Returns:
        The selected row as a mapping.

    Raises:
        InvalidSourceRecord:
            If coordinates do not identify a row.
    """
    if row_group < 0 or row_index < 0:
        raise InvalidSourceRecord("source row coordinates must be non-negative")
    offset = 0
    for batch in parquet.iter_batches(
        row_groups=[row_group], columns=list(columns), batch_size=1
    ):
        _check_batch_bytes(batch, max_batch_bytes)
        if offset == row_index:
            rows = batch.to_pylist()
            if rows:
                return rows[0]
        offset += 1
    raise InvalidSourceRecord("source row coordinate is outside its row group")


def _check_batch_bytes(batch: object, maximum: int) -> None:
    size = int(getattr(batch, "nbytes", 0))
    if size > maximum:
        raise SourceObjectTooLarge(
            f"projected Arrow batch ({size} bytes) exceeds {maximum} bytes"
        )


def _scalar_metadata(row: c.Mapping[str, object]) -> tuple[tuple[str, str], ...]:
    allowed = {
        "broadcaster",
        "creator_affiliation",
        "duration",
        "duration_ms",
        "genre",
        "genre_sub",
        "language",
        "month",
        "platform",
        "show",
        "speaker_count",
        "split",
        "title",
        "year",
    }
    result: list[tuple[str, str]] = []
    for key, value in row.items():
        if key.casefold() in allowed and (
            isinstance(value, (str, int, float, bool)) or value is None
        ):
            result.append((key, json.dumps(value, ensure_ascii=False, sort_keys=True)))
    return tuple(sorted(result))


def _transcript_metadata_columns(names: c.Sequence[str]) -> list[str]:
    excluded = {
        "audio",
        "text",
        "transcript",
        "transcript_text",
        "words",
        "word_timestamps",
        "timestamps",
        "segments",
    }
    return [name for name in names if name.casefold() not in excluded]


def _transcript_payload_columns(names: c.Sequence[str]) -> list[str]:
    wanted = {
        "file_id",
        "audio_id",
        "text",
        "transcript",
        "transcript_text",
        "words",
        "word_timestamps",
        "timestamps",
    }
    return [name for name in names if name.casefold() in wanted]


def _validate_revision(revision: str) -> None:
    """Reject mutable Hub refs and abbreviated commit identifiers.

    Raises:
        ValueError:
            If ``revision`` is not a complete commit SHA.
    """
    if not isinstance(revision, str) or _SHA_PATTERN.fullmatch(revision) is None:
        raise ValueError("source revisions must be complete 40-character commit SHAs")


def parse_audio_row(
    row: c.Mapping[str, object],
    expected_file_id: str | None = None,
    default_sampling_rate: int | None = None,
    default_channels: int | None = None,
    max_decoded_audio_bytes: int = _DEFAULT_MAX_DECODED_AUDIO_BYTES,
) -> ParsedAudio:
    """Parse an embedded audio row and expose its source sampling rate.

    Returns:
        The payload and source sampling rate, without resampling it.

    Raises:
        InvalidSourceRecord:
            If the identifier, payload, or sampling rate is invalid.
    """
    file_id = _file_id(row)
    if file_id is None or (
        expected_file_id is not None and file_id != expected_file_id
    ):
        raise InvalidSourceRecord("audio row has no file_id or is inconsistent")
    audio = row.get("audio")
    if not isinstance(audio, c.Mapping):
        raise InvalidSourceRecord(f"audio row {file_id} has no audio mapping")
    declared_rate = audio.get(
        "sampling_rate", row.get("sampling_rate", default_sampling_rate)
    )
    declared_channels = audio.get("channels", row.get("channels", default_channels))
    for name, value in (
        ("sampling rate", declared_rate),
        ("channel", declared_channels),
    ):
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, numbers.Integral)
            or int(value) <= 0
        ):
            raise InvalidSourceRecord(f"audio row {file_id} has an invalid {name}")
    value = audio.get("array")
    if value is None:
        value = audio.get("bytes")
    if value is None:
        value = audio.get("path")
    if value is None or not isinstance(
        value, (bytes, bytearray, memoryview, str, Path, np.ndarray, list, tuple)
    ):
        raise InvalidSourceRecord(f"audio row {file_id} has an invalid payload")
    if isinstance(value, (bytes, bytearray, memoryview)):
        decoded, decoded_rate, actual_channels = _decode_compressed_audio(
            payload=bytes(value),
            file_id=file_id,
            declared_rate=declared_rate,
            declared_channels=declared_channels,
            max_decoded_audio_bytes=max_decoded_audio_bytes,
        )
        return ParsedAudio(
            file_id=file_id,
            value=decoded,
            sampling_rate=decoded_rate,
            channels=actual_channels,
        )
    if declared_rate is None:
        raise InvalidSourceRecord(f"audio row {file_id} has no sampling rate")
    rate = int(declared_rate)
    if isinstance(value, (list, tuple)):
        value = np.asarray(value)
    if isinstance(value, np.ndarray):
        if value.ndim == 1:
            actual_channels = 1
        elif value.ndim == 2:
            actual_channels = int(value.shape[1])
        else:
            raise InvalidSourceRecord(f"audio row {file_id} has an invalid shape")
        if value.shape[0] == 0 or actual_channels == 0:
            raise InvalidSourceRecord("audio payload is empty")
        if declared_channels is not None and actual_channels != int(declared_channels):
            raise InvalidSourceRecord(
                f"audio row {file_id} channel metadata does not match"
            )
    else:
        actual_channels = int(declared_channels or 1)
    if isinstance(value, str):
        value = Path(value)
    return ParsedAudio(
        file_id=file_id, value=value, sampling_rate=rate, channels=actual_channels
    )


def _decode_compressed_audio(
    *,
    payload: bytes,
    file_id: str,
    declared_rate: object,
    declared_channels: object,
    max_decoded_audio_bytes: int,
) -> tuple[np.ndarray, int, int]:
    """Inspect, bound, and decode a compressed audio payload.

    Returns:
        Decoded float32 PCM, its sampling rate, and its channel count.

    Raises:
        ValueError:
            If the decoded-audio limit is not positive.
        DecodedAudioTooLarge:
            If the predicted or actual PCM allocation exceeds the limit.
        InvalidSourceRecord:
            If the payload or its header is malformed or inconsistent.
    """
    if max_decoded_audio_bytes <= 0:
        raise ValueError("max_decoded_audio_bytes must be positive")
    stream = io.BytesIO(payload)
    try:
        info = sf.info(stream)
    except Exception as exc:
        raise InvalidSourceRecord("audio payload is unsupported or corrupt") from exc
    frames = _header_positive_int(info.frames, "frame count", file_id)
    sampling_rate = _header_positive_int(info.samplerate, "sampling rate", file_id)
    channels = _header_positive_int(info.channels, "channel count", file_id)
    declared_rate_int = (
        int(t.cast(numbers.Integral, declared_rate))
        if declared_rate is not None
        else None
    )
    declared_channels_int = (
        int(t.cast(numbers.Integral, declared_channels))
        if declared_channels is not None
        else None
    )
    if declared_rate_int is not None and sampling_rate != declared_rate_int:
        raise InvalidSourceRecord("audio sampling-rate metadata does not match")
    if declared_channels_int is not None and channels != declared_channels_int:
        raise InvalidSourceRecord("audio channel metadata does not match")

    dtype = np.dtype("float32")
    expected_shape = (frames, channels)
    allocation = _pcm_allocation_bytes(
        shape=expected_shape, dtype=dtype, file_id=file_id
    )
    if allocation > max_decoded_audio_bytes:
        raise DecodedAudioTooLarge(
            f"audio row {file_id} decoded PCM exceeds the configured limit"
        )
    try:
        stream.seek(0)
        decoded, decoded_rate = sf.read(stream, dtype=dtype, always_2d=True)
    except Exception as exc:
        raise InvalidSourceRecord("audio payload is unsupported or corrupt") from exc
    actual_rate = _header_positive_int(decoded_rate, "decoded sampling rate", file_id)
    if actual_rate != sampling_rate:
        raise InvalidSourceRecord("decoded sampling rate does not match its header")
    if (
        not isinstance(decoded, np.ndarray)
        or decoded.dtype != dtype
        or decoded.shape != expected_shape
    ):
        raise InvalidSourceRecord("audio payload shape does not match its header")
    actual_allocation = _pcm_allocation_bytes(
        shape=decoded.shape, dtype=decoded.dtype, file_id=file_id
    )
    if actual_allocation > max_decoded_audio_bytes:
        raise DecodedAudioTooLarge(
            f"audio row {file_id} decoded PCM exceeds the configured limit"
        )
    if not np.isfinite(decoded).all():
        raise InvalidSourceRecord("audio payload is empty or non-finite")
    if decoded.shape[0] == 0 or decoded.shape[1] == 0:
        raise InvalidSourceRecord("audio payload is empty or non-finite")
    return np.asarray(decoded, dtype=dtype), actual_rate, channels


def _header_positive_int(value: object, name: str, file_id: str) -> int:
    """Validate a positive finite integer from a libsndfile header.

    Returns:
        The validated integer value.

    Raises:
        InvalidSourceRecord:
            If the header value is not a positive finite integer.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise InvalidSourceRecord(f"audio row {file_id} has an invalid {name}")
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise InvalidSourceRecord(f"audio row {file_id} has an invalid {name}")
    integer_value = int(numeric_value)
    if integer_value != numeric_value or integer_value <= 0:
        raise InvalidSourceRecord(f"audio row {file_id} has an invalid {name}")
    return integer_value


def _pcm_allocation_bytes(
    *, shape: tuple[int, ...], dtype: np.dtype[np.generic], file_id: str
) -> int:
    """Calculate a PCM allocation with bounds safe for NumPy dimensions.

    Returns:
        The allocation size in bytes.

    Raises:
        DecodedAudioTooLarge:
            If the shape cannot be represented by NumPy dimensions.
    """
    limit = np.iinfo(np.intp).max
    allocation = 1
    for dimension in (*shape, dtype.itemsize):
        if dimension <= 0 or allocation > limit // dimension:
            raise DecodedAudioTooLarge(
                f"audio row {file_id} has an unrepresentable decoded allocation"
            )
        allocation *= dimension
    return allocation


def parse_transcript_row(
    *,
    row: c.Mapping[str, object],
    expected_file_id: str | None = None,
    programme_duration_ms: int | None = None,
) -> ParsedTranscript:
    """Parse the P1 transcript schema and reconstruct verbatim text.

    Spacing and untimed records contribute their literal text but do not become
    timing anchors. Their source characters are owned by the following timed word,
    or by the final timed word when they trail the transcript. Lexical ownership is
    accepted only when bounded lookahead finds a speaker-consistent timed anchor.
    Timestamp seconds are rounded to the nearest millisecond using decimal input,
    avoiding binary-float truncation at boundaries.

    Args:
        row:
            A single transcript row from the source dataset.
        expected_file_id (optional):
            Identifier required by the caller's pointer.
        programme_duration_ms (optional):
            If provided, the upper bound for every word end time.

    Returns:
        The exact transcript text and validated timed words.

    Raises:
        InvalidSourceRecord:
            If the row or one of its timed words is malformed.
        InvalidSourceTimestamp:
            If timed words cannot form a non-overlapping timeline.
        TranscriptOverAudio:
            If a timed word endpoint exceeds ``programme_duration_ms``.
    """
    file_id = _file_id(row)
    if file_id is None or (
        expected_file_id is not None and file_id != expected_file_id
    ):
        raise InvalidSourceRecord("transcript file_id is missing or inconsistent")
    if programme_duration_ms is not None and (
        isinstance(programme_duration_ms, bool)
        or not isinstance(programme_duration_ms, int)
        or programme_duration_ms <= 0
    ):
        raise InvalidSourceRecord("programme duration must be positive")
    raw_words = _first_present(row, ("words", "word_timestamps", "timestamps"))
    if not isinstance(raw_words, c.Iterable) or isinstance(
        raw_words, (str, bytes, dict)
    ):
        raise InvalidSourceRecord(f"transcript {file_id} has no word records")
    words: list[SourceWord] = []
    pieces: list[str] = []
    source_cursor = 0
    timed_source_starts: list[int] = []
    previous_end = 0
    previous_anchor: _TimedOwnershipAnchor | None = None
    pending_lexical: list[_PendingLexicalRecord] = []
    zero_duration_tokens_omitted = 0
    untimed_tokens_owned = 0
    ambiguous_source_text_records = 0
    for position, raw in enumerate(raw_words):
        if not isinstance(raw, c.Mapping):
            raise InvalidSourceRecord(f"word {position} is not a mapping")
        text_value = _first_present(raw, ("text", "word", "token"))
        kind = str(
            raw.get("type", raw.get("kind", raw.get("word_type", "")))
        ).casefold()
        is_spacing = (
            kind in {"space", "spacing", "whitespace", "punctuation_spacing"}
            or raw.get("is_space") is True
        )
        if text_value is None and is_spacing:
            pieces.append(" ")
            source_cursor += 1
            untimed_tokens_owned += 1
            continue
        if not isinstance(text_value, str):
            raise InvalidSourceRecord(f"word {position} has invalid text")
        if is_spacing and text_value == "<space>":
            text_value = " "
        source_start = source_cursor
        pieces.append(text_value)
        source_cursor += len(text_value)
        if is_spacing or not text_value.strip():
            untimed_tokens_owned += 1
            continue
        speaker_id = _speaker_id(raw=raw, position=position)
        raw_start = _first_present(raw, ("start_ms", "start"))
        raw_end = _first_present(raw, ("end_ms", "end"))
        if raw_start is None and raw_end is None:
            untimed_tokens_owned += 1
            if _contains_lexical_text(text_value):
                pending_lexical.append(
                    _PendingLexicalRecord(speaker_id=speaker_id, point_ms=None)
                )
            continue
        start = _milliseconds(raw, "start", position)
        end = _milliseconds(raw, "end", position)
        if start < 0 or end < start:
            raise InvalidSourceTimestamp(f"word {position} has an invalid span")
        if end == start:
            zero_duration_tokens_omitted += 1
            untimed_tokens_owned += 1
            if _contains_lexical_text(text_value):
                pending_lexical.append(
                    _PendingLexicalRecord(speaker_id=speaker_id, point_ms=start)
                )
            continue
        if start < previous_end:
            raise InvalidSourceTimestamp(f"word {position} has an invalid span")
        if programme_duration_ms is not None and end > programme_duration_ms:
            raise TranscriptOverAudio(f"word {position} lies outside source audio")
        anchor = _TimedOwnershipAnchor(
            speaker_id=speaker_id, start_ms=start, end_ms=end
        )
        ambiguous_source_text_records += sum(
            not _following_ownership_is_consistent(
                record=record, previous=previous_anchor, following=anchor
            )
            for record in pending_lexical
        )
        pending_lexical.clear()
        previous_anchor = anchor
        previous_end = end
        timed_source_starts.append(source_start)
        words.append(
            TranscriptWord(
                text=text_value, start_ms=start, end_ms=end, speaker_id=speaker_id
            )
        )
    ambiguous_source_text_records += sum(
        not _terminal_ownership_is_consistent(record=record, previous=previous_anchor)
        for record in pending_lexical
    )
    explicit_text = _first_present(row, ("transcript_text", "transcript", "text"))
    if explicit_text is not None and not isinstance(explicit_text, str):
        raise InvalidSourceRecord("transcript text is not a string")
    reconstructed_text = "".join(pieces)
    text = explicit_text if isinstance(explicit_text, str) else reconstructed_text
    if explicit_text is not None and text != reconstructed_text:
        raise InvalidSourceRecord(
            f"transcript {file_id} text does not match its source records"
        )
    if not text:
        raise InvalidSourceRecord(f"transcript {file_id} is empty")
    try:
        annotated_words = annotate_source_words(
            words, text, expected_starts=timed_source_starts
        )
    except ValueError as exc:
        raise InvalidSourceRecord(
            f"transcript {file_id} words do not reconstruct its verbatim text: {exc}"
        ) from exc
    return ParsedTranscript(
        file_id=file_id,
        text=text,
        words=annotated_words,
        zero_duration_tokens_omitted=zero_duration_tokens_omitted,
        untimed_tokens_owned=untimed_tokens_owned,
        ambiguous_source_text_records=ambiguous_source_text_records,
    )


def _contains_lexical_text(value: str) -> bool:
    """Return whether text contains characters beyond punctuation and separators."""
    return any(
        not unicodedata.category(character).startswith(("P", "Z"))
        for character in value
    )


def _first_present(row: c.Mapping[str, object], names: c.Sequence[str]) -> object:
    for name in names:
        if name in row:
            return row[name]
    return None


def _following_ownership_is_consistent(
    *,
    record: _PendingLexicalRecord,
    previous: _TimedOwnershipAnchor | None,
    following: _TimedOwnershipAnchor,
) -> bool:
    """Return whether following-word ownership agrees with speaker and time evidence."""
    if record.point_ms is not None and (
        record.point_ms > following.start_ms
        or (previous is not None and record.point_ms < previous.end_ms)
    ):
        return False
    if record.speaker_id is not None:
        return record.speaker_id == following.speaker_id
    return (
        previous is not None
        and previous.speaker_id is not None
        and previous.speaker_id == following.speaker_id
    )


def _milliseconds(row: c.Mapping[str, object], name: str, position: int) -> int:
    value = row.get(f"{name}_ms")
    if value is None:
        value = row.get(name)
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise InvalidSourceTimestamp(f"word {position} has invalid {name} seconds")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise InvalidSourceTimestamp(f"word {position} has non-finite {name}")
        return int(
            (decimal.Decimal(str(value)) * 1000).quantize(
                decimal.Decimal("1"), rounding=decimal.ROUND_HALF_UP
            )
        )
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise InvalidSourceTimestamp(f"word {position} has invalid {name}_ms")
    return int(value)


def _speaker_id(*, raw: c.Mapping[str, object], position: int) -> str | None:
    """Return one validated source speaker identifier.

    Raises:
        InvalidSourceRecord:
            If the speaker ID is not a string or integer.
    """
    speaker = raw.get("speaker_id", raw.get("speaker"))
    if speaker is not None and (
        isinstance(speaker, bool) or not isinstance(speaker, (str, int))
    ):
        raise InvalidSourceRecord(f"word {position} has an invalid speaker ID")
    return None if speaker is None else str(speaker)


def _terminal_ownership_is_consistent(
    *, record: _PendingLexicalRecord, previous: _TimedOwnershipAnchor | None
) -> bool:
    """Return whether final-word ownership agrees with speaker and time evidence."""
    return (
        previous is not None
        and record.speaker_id is not None
        and record.speaker_id == previous.speaker_id
        and (record.point_ms is None or record.point_ms >= previous.end_ms)
    )


__all__ = [
    "AudioPointer",
    "HfP1Source",
    "InvalidSourceRecord",
    "InvalidSourceTimestamp",
    "P1Source",
    "TranscriptOverAudio",
    "ParsedAudio",
    "ParsedTranscript",
    "SourceError",
    "SourceObjectTooLarge",
    "SourcePlan",
    "SourceSelectionError",
    "SourceShard",
    "TranscriptIndexRejection",
    "TranscriptPointer",
    "TranscriptPointerIndex",
    "TranscriptWord",
    "parse_audio_row",
    "parse_transcript_row",
]
