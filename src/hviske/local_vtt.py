"""Manifest creation and lazy loading for local WAV/VTT corpora."""

import hashlib
import json
import re
import typing as t
from pathlib import Path

import soundfile as sf
import torch
from datasets import Dataset


class Cue(t.TypedDict):
    """A parsed WebVTT cue."""

    start: float
    end: float
    text: str


_TIMESTAMP_PATTERN = re.compile(
    r"^(?:(?P<hours>\d+):)?(?P<minutes>\d{2}):(?P<seconds>\d{2})[.,](?P<millis>\d{3})$"
)


def build_vtt_manifest(
    source_directories: list[Path], output_path: Path, language: str
) -> None:
    """Build a compact JSONL manifest from matching WAV and VTT files.

    Args:
        source_directories:
            Directories containing programme/video WAV and VTT pairs. Subdirectories
            are searched recursively.
        output_path:
            Destination JSONL manifest. Audio is never copied there.
        language:
            ISO language code stored on every cue.

    Raises:
        ValueError:
            If a VTT cue is malformed, empty, or outside its WAV file.
        FileNotFoundError:
            If a WAV file has no matching VTT file.
    """
    rows: list[dict[str, str | float]] = []
    for directory in source_directories:
        for wav_path in sorted(directory.rglob("*.wav")):
            vtt_path = wav_path.with_suffix(".vtt")
            if not vtt_path.is_file():
                raise FileNotFoundError(f"No matching VTT file for {wav_path}")
            audio_duration = float(sf.info(wav_path).duration)
            for cue in parse_vtt(vtt_path):
                if cue["end"] > audio_duration:
                    raise ValueError(f"Cue extends beyond WAV duration: {vtt_path}")
                rows.append(
                    _manifest_row(
                        wav_path=wav_path,
                        start=cue["start"],
                        end=cue["end"],
                        text=cue["text"],
                        language=language,
                    )
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as manifest_file:
        for row in rows:
            manifest_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_vtt(path: Path) -> list[Cue]:
    """Parse non-empty cues from a WebVTT file.

    Args:
        path:
            WebVTT file to parse.

    Returns:
        Cue dictionaries containing ``start``, ``end`` and ``text``.

    Raises:
        ValueError:
            If a cue is malformed or has no text.
    """
    content = path.read_text(encoding="utf-8-sig")
    blocks = re.split(r"\n\s*\n", content.replace("\r\n", "\n"))
    cues: list[Cue] = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines()]
        if not lines or lines[0].startswith(("WEBVTT", "NOTE", "STYLE", "REGION")):
            continue
        timing_lines = [line for line in lines if "-->" in line]
        if len(timing_lines) != 1:
            raise ValueError(f"Malformed VTT cue in {path}")
        timing = timing_lines[0].split("-->", maxsplit=1)
        start = _parse_timestamp(timing[0].strip(), path)
        end = _parse_timestamp(timing[1].split(maxsplit=1)[0], path)
        text_lines = lines[lines.index(timing_lines[0]) + 1 :]
        text = " ".join(text_lines).strip()
        if not text or end <= start:
            raise ValueError(f"Empty or invalid VTT cue in {path}")
        cues.append({"start": start, "end": end, "text": text})
    return cues


def load_vtt_manifest(
    manifest_path: Path, min_seconds: float, max_seconds: float
) -> Dataset:
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
        An Arrow dataset containing metadata only.

    Raises:
            ValueError:
                If a manifest row has an invalid duration.
    """
    rows: list[dict[str, t.Any]] = []
    with manifest_path.open(encoding="utf-8") as manifest_file:
        for line_number, line in enumerate(manifest_file, start=1):
            row = json.loads(line)
            duration = float(row["duration"])
            if min_seconds < duration < max_seconds:
                rows.append(row)
            elif duration < 0:
                raise ValueError(f"Negative duration on manifest line {line_number}")
    return Dataset.from_list(rows)


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
    if audio_array.shape[0] != frame_count:
        raise ValueError(f"Cue is outside source WAV: {path}")
    if audio_array.ndim == 2:
        audio_array = audio_array.mean(axis=1)
    if source_rate != sampling_rate:
        audio_array = torch.from_numpy(audio_array).unsqueeze(0).float()
        audio_array = torch.nn.functional.interpolate(
            audio_array.unsqueeze(0),
            size=round(audio_array.shape[-1] * sampling_rate / source_rate),
            mode="linear",
            align_corners=False,
        )[0, 0].numpy()
    example["audio"] = {"array": audio_array, "sampling_rate": sampling_rate}
    return example


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
