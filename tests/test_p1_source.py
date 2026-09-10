"""Production-shaped tests for bounded P1 source access."""

from __future__ import annotations

import contextlib
import io
import sqlite3
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from hviske.p1_source import (
    HfP1Source,
    InvalidSourceRecord,
    InvalidSourceTimestamp,
    SourceObjectTooLarge,
    SourceShard,
    parse_audio_row,
    parse_transcript_row,
)


def test_audio_parser_accepts_array_and_sampling_rate_metadata() -> None:
    """Array-form audio rows retain a usable numeric payload and source rate."""
    audio = parse_audio_row(
        {"file_id": "x", "audio": {"array": [0.0, 1.0], "sampling_rate": 48_000}}
    )
    assert isinstance(audio.value, np.ndarray)
    assert audio.sampling_rate == 48_000


def test_audio_parser_decodes_genuine_48khz_flac_native_shape() -> None:
    """Embedded FLAC is decoded at its declared rate before later processing."""
    native = np.column_stack(
        (np.linspace(-1.0, 1.0, 96), np.linspace(1.0, -1.0, 96))
    ).astype(np.float32)
    payload = io.BytesIO()
    sf.write(payload, native, 48_000, format="FLAC", subtype="PCM_16")

    audio = parse_audio_row(
        {
            "file_id": "x",
            "audio": {
                "bytes": payload.getvalue(),
                "sampling_rate": 48_000,
                "channels": 2,
            },
        }
    )

    assert isinstance(audio.value, np.ndarray)
    assert audio.value.shape == native.shape
    assert audio.sampling_rate == 48_000
    assert audio.channels == 2
    assert np.allclose(audio.value[:, 0], native[:, 0], atol=1 / 32_768)


def test_audio_parser_derives_flac_metadata_when_declarations_are_absent() -> None:
    """FLAC headers are authoritative when optional row metadata is absent."""
    payload = io.BytesIO()
    sf.write(payload, np.zeros((12, 1), dtype=np.float32), 22_050, format="FLAC")

    parsed = parse_audio_row({"file_id": "x", "audio": {"bytes": payload.getvalue()}})

    assert parsed.sampling_rate == 22_050
    assert parsed.channels == 1
    assert isinstance(parsed.value, np.ndarray)


def test_discovery_projects_out_audio_column(tmp_path: Path) -> None:
    """Metadata discovery returns no embedded audio payload."""
    _write_source_files(tmp_path)
    source = HfP1Source(local_root=tmp_path)
    shard = source.plan(
        audio_revision="a" * 40, transcript_revision="b" * 40
    ).audio_shards[0]
    rows = list(source.iter_programme_metadata(shard=shard))
    assert [row["file_id"] for row in rows] == ["programme-b", "programme-a"]
    assert all("audio" not in row for row in rows)
    pointers = list(source.iter_programme_pointers(shard=shard))
    assert [pointer.file_id for pointer in pointers] == ["programme-b", "programme-a"]


def _write_source_files(root: Path) -> tuple[Path, Path]:
    """Write small Parquet shards with the P1 audio and transcript schemas.

    Returns:
        The generated audio and transcript paths.
    """
    audio_root = root / "syvai" / "p1" / "data"
    transcript_root = root / "syvai" / "p1-transcripts" / "data"
    audio_root.mkdir(parents=True)
    transcript_root.mkdir(parents=True)
    audio_rows = [
        {
            "file_id": "programme-b",
            "title": "B",
            "duration_ms": 1_000,
            "audio": {"bytes": b"b", "sampling_rate": 48_000},
        },
        {
            "file_id": "programme-a",
            "title": "A",
            "duration_ms": 1_000,
            "audio": {"bytes": b"a", "sampling_rate": 48_000},
        },
    ]
    transcript_rows = [
        {
            "file_id": "programme-a",
            "speaker_count": 1,
            "transcript_text": "Hej verden",
            "words": [
                {"word": "Hej", "start": 0.001, "end": 0.251, "speaker": "s1"},
                {"word": " ", "type": "spacing"},
                {"word": "verden", "start": 0.251, "end": 0.501, "speaker": "s2"},
            ],
        },
        {
            "file_id": "programme-b",
            "speaker_count": 2,
            "transcript_text": "God dag",
            "words": [
                {"word": "God", "start": 0.0, "end": 0.2, "speaker": "s4"},
                {"word": " ", "type": "spacing"},
                {"word": "dag", "start": 0.2, "end": 0.4, "speaker": "s5"},
            ],
        },
    ]
    audio_path = audio_root / "train-z.parquet"
    transcript_path = transcript_root / "train.parquet"
    pq.write_table(pa.Table.from_pylist(audio_rows), audio_path, row_group_size=1)
    pq.write_table(
        pa.Table.from_pylist(transcript_rows), transcript_path, row_group_size=1
    )
    return audio_path, transcript_path


def test_hf_source_reads_under_explicit_dataset_namespace() -> None:
    """Pinned remote source reads use the datasets namespace explicitly."""
    source = HfP1Source()
    paths: list[str] = []

    class Filesystem:
        def open(self, path: str, *_: object, **__: object) -> io.BytesIO:
            paths.append(path)
            return io.BytesIO(b"not parquet")

    source._fs = Filesystem()
    with pytest.raises(Exception):
        with source._parquet(
            "data/train.parquet", repository="org/source", revision="a" * 40
        ):
            pass
    assert paths == ["datasets/org/source/data/train.parquet"]


def test_metadata_batches_are_capped_by_rows_and_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Projected discovery honours both the row cap and hard byte cap."""
    calls: list[dict[str, object]] = []

    class Batch:
        nbytes = 32

        def to_pylist(self) -> list[dict[str, object]]:
            return [{"file_id": "programme-a"}]

    class Schema:
        names = ("file_id", "audio")

    class Parquet:
        schema_arrow = Schema()
        num_row_groups = 1

        def iter_batches(self, **kwargs: object) -> object:
            calls.append(kwargs)
            return iter((Batch(),))

    @contextlib.contextmanager
    def parquet(_path: str, **_: object) -> object:
        yield Parquet()

    source = HfP1Source(max_batch_rows=2, max_batch_bytes=64)
    monkeypatch.setattr(source, "_parquet", parquet)
    shard = SourceShard("audio.parquet", 10)
    assert list(source.iter_programme_metadata(shard=shard, batch_size=10))
    assert calls == [{"row_groups": [0], "columns": ("file_id",), "batch_size": 2}]
    calls.clear()
    assert list(source.iter_programme_pointers(shard=shard, batch_size=10))
    assert calls == [{"row_groups": [0], "columns": ("file_id",), "batch_size": 2}]

    class OversizedBatch(Batch):
        nbytes = 65

    class OversizedParquet(Parquet):
        def iter_batches(self, **kwargs: object) -> object:
            calls.append(kwargs)
            return iter((OversizedBatch(),))

    @contextlib.contextmanager
    def oversized_parquet(_path: str, **_: object) -> object:
        yield OversizedParquet()

    monkeypatch.setattr(source, "_parquet", oversized_parquet)
    with pytest.raises(SourceObjectTooLarge):
        list(source.iter_programme_metadata(shard=shard, batch_size=10))


def test_plan_uses_local_tree_metadata_and_orders_shards(tmp_path: Path) -> None:
    """Planning lists objects without opening or iterating any Parquet row."""
    _write_source_files(tmp_path)
    source = HfP1Source(local_root=tmp_path)
    plan = source.plan(audio_revision="a" * 40, transcript_revision="b" * 40)
    assert [shard.path for shard in plan.audio_shards] == ["data/train-z.parquet"]
    assert plan.audio_shards[0].byte_size > 0
    assert plan.transcript_objects[0][0] == "data/train.parquet"


def test_pointer_index_is_disk_backed_and_contains_no_transcript_payload(
    tmp_path: Path,
) -> None:
    """The index retains coordinates and strata, never words or transcript text."""
    _write_source_files(tmp_path)
    source = HfP1Source(local_root=tmp_path)
    plan = source.plan(audio_revision="a" * 40, transcript_revision="b" * 40)
    index = source.build_transcript_index(
        revision=plan.transcript_revision,
        path=tmp_path / "pointers.sqlite",
        objects=plan.transcript_objects,
    )
    pointer = index.get("programme-a")
    assert pointer is not None
    assert pointer.remote_row_pointer == ("data/train.parquet", 0, 0)
    assert dict(pointer.metadata)["speaker_count"] == "1"
    with sqlite3.connect(tmp_path / "pointers.sqlite") as connection:
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(transcript_pointers)")
        }
    assert columns == {
        "file_id",
        "path",
        "row_group",
        "row_index",
        "revision",
        "byte_size",
        "metadata",
    }
    assert len(index) == 2
    fetched = source.fetch_transcript(pointer)
    assert fetched.text == "Hej verden"
    assert [word.speaker_id for word in fetched.words] == ["s1", "s2"]
    assert fetched.words[0].start_ms == 1
    assert fetched.words[0].end_ms == 251


def test_source_object_cap_is_checked_before_audio_open(tmp_path: Path) -> None:
    """A rejected object cannot reach the Parquet reader."""
    _write_source_files(tmp_path)
    source = HfP1Source(local_root=tmp_path, max_source_object_bytes=1)
    shard = SourceShard(path="data/train-z.parquet", byte_size=2, revision="a" * 40)
    with pytest.raises(SourceObjectTooLarge):
        list(source.iter_programme_metadata(shard=shard))


def test_targeted_audio_selection_reads_one_48khz_row(tmp_path: Path) -> None:
    """An audio pointer selects only its row and exposes the source rate."""
    _write_source_files(tmp_path)
    source = HfP1Source(local_root=tmp_path)
    shard = source.plan(
        audio_revision="a" * 40, transcript_revision="b" * 40
    ).audio_shards[0]
    pointer = next(
        pointer
        for pointer in source.iter_programme_pointers(shard=shard)
        if pointer.file_id == "programme-a"
    )
    audio = source.fetch_audio(pointer=pointer)
    assert audio.file_id == "programme-a"
    assert audio.value == b"a"
    assert audio.sampling_rate == 48_000


def test_targeted_transcript_index_fails_if_pointer_is_absent(tmp_path: Path) -> None:
    """A missing targeted transcript is an explicit selection failure."""
    _write_source_files(tmp_path)
    source = HfP1Source(local_root=tmp_path)
    plan = source.plan(audio_revision="a" * 40, transcript_revision="b" * 40)
    with pytest.raises(ValueError, match="no valid transcript pointer"):
        source.build_transcript_index(
            revision=plan.transcript_revision,
            path=tmp_path / "targeted-pointers.sqlite",
            objects=plan.transcript_objects,
            source_file_id="programme-missing",
        )


def test_targeted_transcript_index_stops_at_first_valid_pointer(tmp_path: Path) -> None:
    """Targeted transcript discovery does not build a corpus-wide index."""
    _write_source_files(tmp_path)
    source = HfP1Source(local_root=tmp_path)
    plan = source.plan(audio_revision="a" * 40, transcript_revision="b" * 40)
    index = source.build_transcript_index(
        revision=plan.transcript_revision,
        path=tmp_path / "targeted-pointers.sqlite",
        objects=plan.transcript_objects,
        source_file_id="programme-a",
    )
    assert len(index) == 1
    assert index.get("programme-a") is not None
    assert index.get("programme-b") is None


def test_transcript_parser_omits_zero_duration_tokens_without_losing_source_text() -> (
    None
):
    """Zero-duration tokens remain in exact separator spans and are counted."""
    parsed = parse_transcript_row(
        row={
            "file_id": "x",
            "words": [
                {"text": "left", "start_ms": 0, "end_ms": 100},
                {"text": "<noise>", "start_ms": 100, "end_ms": 100},
                {"text": "right", "start_ms": 100, "end_ms": 200},
            ],
        }
    )

    assert parsed.text == "left<noise>right"
    assert [word.text for word in parsed.words] == ["left", "right"]
    assert parsed.zero_duration_tokens_omitted == 1
    assert parsed.words[1].separator_text == "<noise>"
    assert parsed.words[1].separator_span is not None
    assert parsed.words[1].separator_span.start == 4
    assert parsed.words[1].separator_span.end == 11


@pytest.mark.parametrize(
    "words",
    [
        [{"text": "bad", "start_ms": -1, "end_ms": 1}],
        [{"text": "bad", "start_ms": 2, "end_ms": 1}],
        [
            {"text": "one", "start_ms": 0, "end_ms": 10},
            {"text": "two", "start_ms": 9, "end_ms": 20},
        ],
    ],
)
def test_transcript_parser_rejects_invalid_timestamp_spans(
    words: list[dict[str, object]],
) -> None:
    """Negative, reversed, and positive-overlap source spans fail explicitly."""
    with pytest.raises(InvalidSourceTimestamp):
        parse_transcript_row(row={"file_id": "x", "words": words})


def test_transcript_parser_rejects_invalid_words_and_preserves_verbatim_text() -> None:
    """Spacing is not a timed word, while invalid word records fail closed."""
    parsed = parse_transcript_row(
        row={
            "file_id": "x",
            "words": [
                {"text": "a", "start": 0.0001, "end": 0.0009, "speaker_id": "left"},
                {"text": " ", "type": "spacing"},
                {"text": "b", "start": 0.001, "end": 0.002, "speaker_id": "right"},
            ],
        }
    )
    assert parsed.text == "a b"
    assert [word.start_ms for word in parsed.words] == [0, 1]
    assert [word.speaker_id for word in parsed.words] == ["left", "right"]
    with pytest.raises(InvalidSourceRecord):
        parse_transcript_row(
            row={"file_id": "x", "words": [{"word": "bad", "start": 1.0}]}
        )
