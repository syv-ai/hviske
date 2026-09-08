"""Focused tests for private-only Hub publication."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from _pytest.monkeypatch import MonkeyPatch

import hviske.utils as utils


class FakeHubApi:
    """Small stateful Hub API double for visibility checks."""

    def __init__(self, private: bool | None) -> None:
        """Initialise the fake repository visibility."""
        self.private = private
        self.create_calls: list[dict[str, object]] = []
        self.info_calls = 0
        self.upload_calls: list[dict[str, object]] = []

    def repo_info(self, **_: object) -> SimpleNamespace:
        """Return repository metadata or the configured missing error.

        Raises:
            FakeRepositoryNotFoundError:
                If the fake repository is missing.
        """
        self.info_calls += 1
        if self.private is None:
            raise FakeRepositoryNotFoundError
        return SimpleNamespace(private=self.private)

    def create_repo(self, **kwargs: object) -> None:
        """Record private repository creation."""
        self.create_calls.append(kwargs)
        self.private = True

    def upload_file(self, **kwargs: object) -> SimpleNamespace:
        """Record the model-card upload.

        Returns:
            A placeholder commit object.
        """
        self.upload_calls.append(kwargs)
        return SimpleNamespace()


class FakeRepositoryNotFoundError(Exception):
    """Hub repository-not-found error for the mocked API."""


def test_private_only_refuses_private_false() -> None:
    """A private-only run cannot be configured as public."""
    with pytest.raises(ValueError, match="private=true"):
        utils.validate_private_only_config({"private_only": True, "private": False})


def test_private_only_refuses_existing_public_repository(
    monkeypatch: MonkeyPatch,
) -> None:
    """An existing public destination is never changed or uploaded to."""
    api = FakeHubApi(private=False)
    monkeypatch.setattr(utils, "HfApi", lambda **_: api)

    with pytest.raises(PermissionError, match="public repository"):
        utils.ensure_private_hub_repository(repo_id="syvai/hviske-v6", token="token")

    assert api.create_calls == []


def test_private_only_creates_missing_repository_as_private(
    monkeypatch: MonkeyPatch,
) -> None:
    """A missing destination is created privately and checked immediately."""
    api = FakeHubApi(private=None)
    monkeypatch.setattr(utils, "HfApi", lambda **_: api)
    monkeypatch.setattr(utils, "RepositoryNotFoundError", FakeRepositoryNotFoundError)

    utils.ensure_private_hub_repository(repo_id="syvai/hviske-v6", token="token")

    assert api.create_calls[0]["private"] is True
    assert api.info_calls == 2
    assert api.private is True


def test_private_only_upload_verifies_visibility_after_upload(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    """The standalone uploader checks visibility after the final Hub commit."""
    api = FakeHubApi(private=True)
    monkeypatch.setattr(utils, "HfApi", lambda **_: api)
    monkeypatch.setattr(utils, "upload_folder", lambda **_: SimpleNamespace())
    monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "token")

    utils.publish_model_folder(
        folder_path=tmp_path,
        repo_id="syvai/hviske-v6",
        finetuned_from="CohereLabs/cohere-transcribe-03-2026",
        private=True,
        model_card_languages=["da", "en"],
    )

    assert api.info_calls == 2
    assert api.upload_calls[0]["path_in_repo"] == "README.md"
    assert b"- da" in api.upload_calls[0]["path_or_fileobj"]
    assert b"- en" in api.upload_calls[0]["path_or_fileobj"]
