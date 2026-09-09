"""Production-shaped tests for bounded P1 source access."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hviske.p1_source import (
    HfP1Source,
    InvalidSourceRecord,
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
