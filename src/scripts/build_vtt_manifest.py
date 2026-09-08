"""Build a zero-copy WAV/VTT manifest for local ASR training."""

import argparse
from pathlib import Path

from hviske.local_vtt import build_vtt_manifest


def main() -> None:
    """Parse arguments and build a local audio manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        action="append",
        type=Path,
        required=True,
        help="Directory containing matching WAV/VTT files; may be repeated.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language", required=True, help="ISO language code.")
    args = parser.parse_args()
    build_vtt_manifest(
        source_directories=args.source_dir,
        output_path=args.output,
        language=args.language,
    )


if __name__ == "__main__":
    main()
