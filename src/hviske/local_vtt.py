"""Manifest creation and lazy loading for local WAV/VTT corpora."""

import dataclasses
import hashlib
import html
import json
import logging
import os
import re
import tempfile
import typing as t
from pathlib import Path

import soundfile as sf
import torch
import torchaudio.functional
from datasets import Features, IterableDataset, Value

logger = logging.getLogger(__package__)


class Cue(t.TypedDict):
    """A parsed WebVTT cue."""

    start: float
    end: float
    text: str


@dataclasses.dataclass
class ManifestSummary:
    """Summary of a manifest build."""

    files_seen: int = 0
    files_written: int = 0
    files_skipped: int = 0
    cues_written: int = 0
    cues_skipped: int = 0


@dataclasses.dataclass
class VTTParseStats:
    """Counters collected while parsing a VTT corpus."""

    cues_skipped: int = 0


_TIMESTAMP_PATTERN = re.compile(
    r"^(?:(?P<hours>\d+):)?(?P<minutes>\d{2}):(?P<seconds>\d{2})[.,](?P<millis>\d{3})$"
)
_INLINE_TIMESTAMP_PATTERN = re.compile(r"<(?:(?:\d{2}:)?\d{2}:\d{2}[.,]\d{3})>")
_MARKUP_PATTERN = re.compile(r"<[^>]*>")


def build_vtt_manifest(
    source_directories: list[Path], output_path: Path, language: str
) -> ManifestSummary:
    """Build a compact JSONL manifest from matching WAV and VTT files.

    The destination is written to a temporary sibling and replaced only after all
    input files have been processed successfully. Malformed VTT files and cues are
    skipped so that one bad caption file cannot stop a corpus build.

    Args:
        source_directories:
            Directories containing programme/video WAV and VTT pairs. Subdirectories
            are searched recursively.
        output_path:
            Destination JSONL manifest. Audio is never copied there.
        language:
            ISO language code stored on every cue.

    Returns:
        Counts of accepted and skipped files and cues.

    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = ManifestSummary()
    parse_stats = VTTParseStats()
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            for directory in source_directories:
                for wav_path in sorted(directory.rglob("*.wav")):
                    summary.files_seen += 1
                    vtt_path = wav_path.with_suffix(".vtt")
                    if not vtt_path.is_file():
                        summary.files_skipped += 1
                        logger.warning("Skipping WAV without VTT: %s", wav_path)
                        continue
                    audio_duration = float(sf.info(wav_path).duration)
                    try:
                        cues = parse_vtt(vtt_path, stats=parse_stats)
                    except (OSError, UnicodeError, ValueError) as error:
                        summary.files_skipped += 1
                        logger.warning("Skipping malformed VTT %s: %s", vtt_path, error)
                        continue
                    file_cues = 0
                    for cue in cues:
                        if cue["end"] > audio_duration:
                            parse_stats.cues_skipped += 1
                            continue
                        row = _manifest_row(
                            wav_path=wav_path,
                            start=cue["start"],
                            end=cue["end"],
                            text=cue["text"],
                            language=language,
                        )
                        temporary_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                        file_cues += 1
                    if file_cues:
                        summary.files_written += 1
                        summary.cues_written += file_cues
                    else:
                        summary.files_skipped += 1
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    summary.cues_skipped = parse_stats.cues_skipped
    logger.info(
        "Built VTT manifest %s: %d files and %d cues accepted; "
        "%d files and %d cues skipped",
        output_path,
        summary.files_written,
        summary.cues_written,
        summary.files_skipped,
        summary.cues_skipped,
    )
    return summary


def _manifest_row(
    wav_path: Path, start: float, end: float, text: str, language: str
) -> dict[str, str | float]:
    stable_input = f"{wav_path.resolve()}\0{start:.3f}\0{end:.3f}\0{text}"
    stable_id = hashlib.sha256(stable_input.encode()).hexdigest()[:24]
    return {
        "source_wav_path": str(wav_path.resolve()),
        "start": start,
        "end": end,
        "text": text,
        "id": stable_id,
        "duration": end - start,
        "language": language,
    }


def parse_vtt(path: Path, stats: VTTParseStats | None = None) -> list[Cue]:
    """Parse usable cues from a WebVTT file.

    Empty or malformed cue blocks are ignored. YouTube's rolling captions are
    reduced to their newly added words, preventing a full caption from being
    emitted repeatedly as separate training examples.

    Args:
        path:
            WebVTT file to parse.
        stats (optional):
            Counters to update for skipped cue blocks.

    Returns:
        Cue dictionaries containing ``start``, ``end`` and cleaned ``text``.

    Raises:
        ValueError:
            If the file has no WebVTT header or cannot be decoded structurally.
    """
    content = path.read_text(encoding="utf-8-sig").replace("\ufeff", "")
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    if not content.lstrip().startswith("WEBVTT"):
        raise ValueError(f"Missing WEBVTT header in {path}")

    blocks = re.split(r"\n\s*\n", content)
    cues: list[Cue] = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines()]
        if not lines:
            continue
        if lines[0].startswith(("WEBVTT", "NOTE", "STYLE", "REGION")):
            continue
        timing_indexes = [index for index, line in enumerate(lines) if "-->" in line]
        if len(timing_indexes) != 1:
            if any("-->" in line for line in lines):
                _skip_cue(stats)
            continue
        timing_index = timing_indexes[0]
        timing = lines[timing_index].split("-->", maxsplit=1)
        try:
            start = _parse_timestamp(timing[0].strip(), path)
            end = _parse_timestamp(timing[1].split(maxsplit=1)[0], path)
        except ValueError:
            _skip_cue(stats)
            continue
        text = _clean_caption_text(" ".join(lines[timing_index + 1 :]))
        text = _remove_rolling_overlap(cues[-1]["text"] if cues else "", text)
        if not text or end <= start:
            _skip_cue(stats)
            continue
        cues.append({"start": start, "end": end, "text": text})
    return cues


def _clean_caption_text(text: str) -> str:
    text = _INLINE_TIMESTAMP_PATTERN.sub("", text)
    text = _MARKUP_PATTERN.sub(" ", text)
    text = html.unescape(text)
    return " ".join(text.split())


def _parse_timestamp(value: str, path: Path) -> float:
    match = _TIMESTAMP_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"Malformed VTT timestamp {value!r} in {path}")
    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes"))
    seconds = int(match.group("seconds"))
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"Malformed VTT timestamp {value!r} in {path}")
    return hours * 3600 + minutes * 60 + seconds + int(match.group("millis")) / 1000


def _remove_rolling_overlap(previous: str, current: str) -> str:
    if not previous or not current:
        return current
    if current == previous:
        return ""
    if current.startswith(previous):
        return current[len(previous) :].strip()
    previous_words = previous.split()
    current_words = current.split()
    max_overlap = min(len(previous_words), len(current_words))
    for overlap in range(max_overlap, 0, -1):
        if previous_words[-overlap:] == current_words[:overlap]:
            return " ".join(current_words[overlap:])
    return current


def _skip_cue(stats: VTTParseStats | None) -> None:
    if stats is not None:
        stats.cues_skipped += 1


def decode_vtt_audio(
    example: dict[str, object], sampling_rate: int
) -> dict[str, object]:
    """Read one cue from its source WAV, rather than copying the source file.

    Args:
        example:
            A manifest row.
        sampling_rate:
            Target sampling rate.

    Returns:
        The row with an in-memory ``audio`` array containing only the cue.

    Raises:
        ValueError:
            If the cue does not fit inside the source WAV.
    """
    path = Path(str(example["source_wav_path"]))
    start = float(t.cast(float, example["start"]))
    end = float(t.cast(float, example["end"]))
    with sf.SoundFile(path) as audio_file:
        source_rate = audio_file.samplerate
        start_frame = round(start * source_rate)
        frame_count = round((end - start) * source_rate)
        audio_file.seek(start_frame)
        audio_array = audio_file.read(frames=frame_count, dtype="float32")
    if start < 0 or end < start or audio_array.shape[0] != frame_count:
        raise ValueError(f"Cue is outside source WAV: {path}")

    waveform = torch.as_tensor(audio_array, dtype=torch.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=1)
    waveform = waveform.contiguous()
    if source_rate != sampling_rate:
        waveform = torchaudio.functional.resample(
            waveform.unsqueeze(0), orig_freq=source_rate, new_freq=sampling_rate
        )[0]
    audio_array = waveform.contiguous().numpy()
    example["audio"] = {"array": audio_array, "sampling_rate": sampling_rate}
    return example


def load_vtt_manifest(
    manifest_path: Path, min_seconds: float, max_seconds: float
) -> IterableDataset:
    """Load manifest rows without decoding any audio.

    Duration filtering happens while the manifest is read, before a WAV is opened for
    sample data. The returned dataset retains source paths and offsets for lazy slicing.

    Args:
        manifest_path:
            JSONL manifest produced by :func:`build_vtt_manifest`.
        min_seconds:
            Exclusive lower duration bound.
        max_seconds:
            Exclusive upper duration bound.

    Returns:
        An iterable dataset containing metadata only.
    """

    def manifest_rows() -> t.Iterator[dict[str, t.Any]]:
        with manifest_path.open(encoding="utf-8") as manifest_file:
            for line_number, line in enumerate(manifest_file, start=1):
                row = json.loads(line)
                duration = float(row["duration"])
                if duration < 0:
                    raise ValueError(
                        f"Negative duration on manifest line {line_number}"
                    )
                if min_seconds < duration < max_seconds:
                    yield row

    return IterableDataset.from_generator(
        generator=manifest_rows,
        features=Features(
            source_wav_path=Value("string"),
            start=Value("float64"),
            end=Value("float64"),
            text=Value("string"),
            id=Value("string"),
            duration=Value("float64"),
            language=Value("string"),
        ),
    )
