"""Publish a saved model to a verified private Hugging Face repository."""

from pathlib import Path

import click

from hviske.utils import publish_model_folder


@click.command()
@click.argument(
    "model_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.argument("repo_id")
@click.option("--finetuned-from", required=True, help="Base model identifier.")
@click.option(
    "--language",
    multiple=True,
    default=("da", "en"),
    show_default=True,
    help="Language metadata to add to the model card.",
)
@click.option(
    "--private",
    "private",
    is_flag=True,
    help="Confirm that this publication must remain private.",
)
@click.option(
    "--training-dataset-id",
    multiple=True,
    help="Exact Hugging Face dataset ID used for training (repeatable).",
)
@click.option(
    "--evaluation-status",
    default="Not evaluated.",
    show_default=True,
    help="Evaluation status to record in the model card.",
)
def main(
    model_dir: Path,
    repo_id: str,
    finetuned_from: str,
    language: tuple[str, ...],
    private: bool,
    training_dataset_id: tuple[str, ...],
    evaluation_status: str,
) -> None:
    """Upload MODEL_DIR to REPO_ID without exposing private training artefacts."""
    publish_model_folder(
        folder_path=model_dir,
        repo_id=repo_id,
        finetuned_from=finetuned_from,
        private=private,
        model_card_languages=list(language),
        training_dataset_ids=list(training_dataset_id),
        evaluation_status=evaluation_status,
    )


if __name__ == "__main__":
    main()
