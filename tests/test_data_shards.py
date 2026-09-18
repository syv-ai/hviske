"""Tests for bounded Hub shard loading and pre-decode shuffling."""

import typing as t

import pytest
from datasets import Dataset

from hviske.data import _load_transcript_dataset, _resolve_hub_data_files

REVISION = "1" * 40


def test_hub_shard_range_rejects_empty_selection() -> None:
    """A valid range still fails when no pinned filename exists."""
    with pytest.raises(ValueError, match="resolved no existing files"):
        _resolve_hub_data_files(
            dataset_id="organisation/dataset",
            revision=REVISION,
            selection={
                "template": "data/train-{shard:05d}.parquet",
                "start": 1,
                "end": 2,
            },
            hub_api=FakeHubApi(["data/train-00000.parquet"]),
        )


class FakeHubApi:
    """Return a fixed repository tree and record pinned lookups."""

    def __init__(self, files: list[str]) -> None:
        """Store the repository-relative files returned by the fake."""
        self.files = files
        self.calls: list[tuple[str, str, str]] = []

    def list_repo_files(
        self, repo_id: str, *, repo_type: str, revision: str
    ) -> list[str]:
        """Return configured repository-relative paths."""
        self.calls.append((repo_id, repo_type, revision))
        return self.files


def test_hub_shard_range_resolves_only_existing_pinned_files() -> None:
    """Unrelated files and a missing in-range shard never reach the loader."""
    api = FakeHubApi(
        [
            "data/train-00000.parquet",
            "data/train-00001.parquet",
            "data/train-00003.parquet",
            "data/train-00004.parquet",
            "data/train-00005.parquet",
            "README.md",
        ]
    )

    resolved = _resolve_hub_data_files(
        dataset_id="organisation/dataset",
        revision=REVISION,
        selection={"template": "data/train-{shard:05d}.parquet", "start": 1, "end": 4},
        hub_api=api,
    )

    assert resolved == [
        "data/train-00001.parquet",
        "data/train-00003.parquet",
        "data/train-00004.parquet",
    ]
    assert resolved is not None
    assert api.calls == [("organisation/dataset", "dataset", REVISION)]
    assert all("?" not in path and "://" not in path for path in resolved)

    calls: list[dict[str, object]] = []

    def loader(**kwargs: object) -> Dataset:
        calls.append(kwargs)
        return Dataset.from_dict({"text": ["one"]})

    _load_transcript_dataset(
        dataset_id="organisation/dataset",
        subset="default",
        split="train",
        revision=REVISION,
        cache_dir=None,
        dataset_loader=loader,
        data_files=resolved,
    )
    assert calls[0]["data_files"] == resolved


@pytest.mark.parametrize(
    ("selection", "revision", "message"),
    [
        (None, None, None),
        (
            {"template": "data/{shard}.parquet", "start": -1, "end": 2},
            REVISION,
            "bounds",
        ),
        (
            {"template": "data/{shard}.parquet", "start": 3, "end": 2},
            REVISION,
            "bounds",
        ),
        (
            {"template": "data/{other}.parquet", "start": 0, "end": 2},
            REVISION,
            "template",
        ),
        (
            {"template": "https://host/{shard}", "start": 0, "end": 2},
            REVISION,
            "template",
        ),
        (
            {"template": "data/{shard}.parquet", "start": 0, "end": 2},
            "main",
            "immutable",
        ),
    ],
)
def test_hub_shard_range_validates_configuration(
    selection: dict[str, object] | None, revision: str | None, message: str | None
) -> None:
    """Malformed templates, bounds, and mutable revisions fail before listing."""
    if selection is None:
        assert (
            _resolve_hub_data_files(
                dataset_id="organisation/dataset",
                revision=revision,
                selection=selection,
                hub_api=FakeHubApi([]),
            )
            is None
        )
        return
    with pytest.raises(ValueError, match=t.cast(str, message)):
        _resolve_hub_data_files(
            dataset_id="organisation/dataset",
            revision=revision,
            selection=selection,
            hub_api=FakeHubApi(["data/0.parquet"]),
        )
