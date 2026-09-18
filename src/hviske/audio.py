"""Audio loading helpers."""

import io
import logging
import shutil
import typing as t
from pathlib import Path
from zipfile import ZipFile

import httpx
import soundfile as sf
import torch
from torch_audiomentations.utils.io import Audio, AudioFile
from tqdm.auto import tqdm

logger = logging.getLogger(__package__)


class SoundfileAudio(Audio):
    """Load audio files with soundfile and resample them on the fly."""

    def __call__(
        self, file: AudioFile, sample_offset: int = 0, num_samples: int | None = None
    ) -> torch.Tensor:
        """Decode an audio path without relying on torchaudio's file backend.

        In-memory sample dictionaries continue to use the implementation supplied by
        torch-audiomentations.  File paths are decoded with soundfile, then use the
        base class's channel mixing and torchaudio resampling implementation.

        Args:
            file:
                A path, an audio-path dictionary, or an in-memory sample dictionary.
            sample_offset (optional):
                Start offset in samples at the target rate. Defaults to 0.
            num_samples (optional):
                Number of samples to return at the target rate. Defaults to the end.

        Returns:
            A channel-first floating-point waveform.
        """
        self.is_valid(file)

        if isinstance(file, dict) and "samples" in file:
            if num_samples is None:
                return super().__call__(file=file, sample_offset=sample_offset)
            return super().__call__(
                file=file, sample_offset=sample_offset, num_samples=num_samples
            )

        if isinstance(file, dict):
            audio_path = t.cast(str | Path, file["audio"])
            channel = t.cast(int | None, file.get("channel"))
        else:
            audio_path = file
            channel = None

        total_num_samples, original_sample_rate = self.get_audio_metadata(audio_path)
        original_sample_offset = round(
            sample_offset * original_sample_rate / self.sample_rate
        )
        if num_samples is None:
            original_num_samples = max(total_num_samples - original_sample_offset, 0)
        else:
            original_num_samples = round(
                num_samples * original_sample_rate / self.sample_rate
            )

        # soundfile raises when a start position is past EOF. Clamping the position
        # lets the requested target-rate length be fulfilled by the padding below.
        read_start = min(max(original_sample_offset, 0), total_num_samples)
        available_num_samples = max(total_num_samples - read_start, 0)
        read_num_samples = min(max(original_num_samples, 0), available_num_samples)
        original_data, _ = sf.read(
            file=audio_path,
            start=read_start,
            stop=read_start + read_num_samples,
            dtype="float32",
            always_2d=True,
        )
        samples = torch.from_numpy(original_data.T)

        if channel is not None:
            samples = samples[channel - 1 : channel, :]

        if samples.shape[-1] == 0:
            if self.mono and samples.shape[0] > 1:
                result = samples.mean(dim=0, keepdim=True)
            else:
                result = samples
        else:
            result = self.downmix_and_resample(
                samples=samples, sample_rate=original_sample_rate
            )

        if num_samples is not None:
            result = result[..., :num_samples]
            if result.shape[-1] < num_samples:
                result = torch.nn.functional.pad(
                    result, (0, num_samples - result.shape[-1])
                )

        return result

    @staticmethod
    def get_audio_metadata(file_path: str | Path) -> tuple[int, int]:
        """Return the number of frames and sample rate for an audio file."""
        info = sf.info(file_path)
        return info.frames, info.samplerate


def download_background_noises() -> None:
    """Download background noises for audio augmentation.

    This function downloads the background noises to the `background-noises` directory,
    and will do nothing if the directory already exists.
    """
    background_noises_path = Path("background-noises")
    if background_noises_path.exists():
        return

    logger.info("Downloading background noises from the ESC-50 dataset...")

    # Download the ESC-50 dataset zip file as a stream
    zip_url = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
    chunks = []
    with httpx.stream(method="GET", url=zip_url, follow_redirects=True) as response:
        for chunk in tqdm(
            response.iter_bytes(),
            desc="Downloading ESC-50 dataset",
            unit="B",
            unit_scale=True,
            total=int(response.headers.get("Content-Length", 0)),
        ):
            chunks.append(chunk)
    content = b"".join(chunks)

    # Unzip only the audio files from the ESC-50 dataset
    with ZipFile(file=io.BytesIO(content)) as zip_file:
        audio_files = [
            file_info
            for file_info in zip_file.infolist()
            if file_info.filename.startswith("ESC-50-master/audio/")
        ]
        zip_file.extractall(members=audio_files, path=background_noises_path)

    # Move audio files to the root of the background-noises directory
    extracted_audio_path = background_noises_path / "ESC-50-master" / "audio"
    for audio_file in extracted_audio_path.iterdir():
        audio_file.rename(background_noises_path / audio_file.name)

    # Remove the extracted directories
    shutil.rmtree(background_noises_path / "ESC-50-master")

    logger.info("Background noises downloaded successfully.")
