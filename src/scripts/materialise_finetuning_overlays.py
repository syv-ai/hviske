"""Materialise pinned positional overlays as durable local Parquet shards."""

import argparse
import logging
import typing as t
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from hviske.materialised_overlays import materialise_finetuning_overlays
from p1_dataset.source import harden_p1_logging

logger = logging.getLogger("hviske_overlay_materialisation")


def main() -> None:
    """Compose a preset and materialise each paired physical overlay shard."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="sparkie_bilingual")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--minimum-free-gib",
        type=float,
        default=10.0,
        help="Free disk reserve retained throughout the build (default: 10 GiB)",
    )
    args = parser.parse_args()
    if args.minimum_free_gib < 0:
        parser.error("--minimum-free-gib must be non-negative")

    # Install URL redaction before configuration resolution or any Hub client exists.
    harden_p1_logging()
    config_dir = Path(__file__).resolve().parents[2] / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name=args.config_name)
    OmegaConf.resolve(config)
    manifest = materialise_finetuning_overlays(
        config=config,
        output_root=args.output_root,
        minimum_free_bytes=int(args.minimum_free_gib * 1024**3),
    )
    sources = t.cast(list[object], manifest["sources"])
    logger.info("Completed %s materialised overlay sources", len(sources))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
