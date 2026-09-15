"""Audio loading helpers."""

from pathlib import Path

import soundfile as sf
from torch_audiomentations.utils.io import Audio


class SoundfileAudio(Audio):
    """Load audio with soundfile metadata and torch-audiomentations decoding."""

    @staticmethod
    def get_audio_metadata(file_path: str | Path) -> tuple[int, int]:
        """Return the number of frames and sample rate for an audio file."""
        info = sf.info(file_path)
        return info.frames, info.samplerate
