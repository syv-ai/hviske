"""Subprocess tests for privacy-safe P1 command-line output."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parents[1]
_FINALISER = _PROJECT_ROOT / "src/scripts/finalise_p1_segments.py"
_SANITY_GATE = _PROJECT_ROOT / "src/scripts/run_p1_v8_sanity_gate.py"
_REVISION = "a" * 40
_DIGEST = "b" * 64


def test_finaliser_failure_is_sanitised(tmp_path: Path) -> None:
    """A finaliser exception cannot leak Hub errors or request content."""
    _write_sitecustomize(
        tmp_path,
        """
import builtins

_original_import = builtins.__import__


def _import(name, globals=None, locals=None, fromlist=(), level=0):
    module = _original_import(name, globals, locals, fromlist, level)
    if name == "p1_dataset.finalisation":
        def fail(**kwargs):
            raise RuntimeError(
                "RAW_HUB_ERROR segment-secret payload-secret "
                "https://evil.example/resolve/commit?token=credential-secret"
            )
        module.finalise_p1_corpus = fail
    return module


builtins.__import__ = _import
""",
    )

    result = _run_cli(
        _FINALISER,
        "--revision",
        _REVISION,
        "--run-root",
        str(tmp_path / "private-run"),
        "--pipeline-config-sha256",
        _DIGEST,
        "--report",
        str(tmp_path / "private-report.json"),
        extra_python_path=tmp_path,
    )

    _assert_safe_failure(
        result,
        "RAW_HUB_ERROR",
        "segment-secret",
        "payload-secret",
        "https://evil.example/resolve/commit",
        "credential-secret",
    )


def _assert_safe_failure(
    result: subprocess.CompletedProcess[str], *secrets: str
) -> None:
    """Ensure a failed CLI does not expose local or remote sensitive values."""
    assert result.returncode == 1
    output = result.stdout + result.stderr
    assert str(_PROJECT_ROOT) not in output
    assert "?" not in output
    for secret in secrets:
        assert secret not in output


def _run_cli(
    script: Path, *arguments: str, extra_python_path: Path | None = None
) -> subprocess.CompletedProcess[str]:
    """Run a P1 CLI in a subprocess with an optional import hook.

    Returns:
        The completed subprocess result.
    """
    environment = os.environ.copy()
    paths = [str(_PROJECT_ROOT / "src")]
    if extra_python_path is not None:
        paths.insert(0, str(extra_python_path))
    if environment.get("PYTHONPATH"):
        paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    return subprocess.run(
        [sys.executable, str(script), *arguments],
        cwd=_PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def _write_sitecustomize(tmp_path: Path, body: str) -> None:
    """Install a subprocess-only import hook for isolated CLI tests."""
    (tmp_path / "sitecustomize.py").write_text(body, encoding="utf-8")


def test_finaliser_help_has_no_import_warning_or_absolute_path() -> None:
    """Finaliser help is successful and does not expose its checkout path."""
    result = _run_cli(_FINALISER, "--help")

    assert result.returncode == 0
    output = result.stdout + result.stderr
    assert "usage:" in result.stdout
    assert 'Field name "schema"' not in output
    assert str(_PROJECT_ROOT) not in output


def test_finaliser_progresses_with_adapter_commit_history(tmp_path: Path) -> None:
    """The CLI supplies its production adapter for non-empty batch evidence."""
    _write_sitecustomize(
        tmp_path,
        f"""
import warnings
from types import SimpleNamespace

warnings.filterwarnings("ignore", message=r'^Field name "schema"')

import p1_dataset.finalisation as finalisation
from p1_dataset.publish import HfApiAdapter


class Api:
    def list_repo_commits(self, **kwargs):
        assert kwargs == {{
            "repo_id": "syvai/p1-segments",
            "repo_type": "dataset",
            "revision": "{_REVISION}",
            "token": True,
            "formatted": False,
        }}
        return [
            SimpleNamespace(commit_id="{_REVISION}"),
            SimpleNamespace(commit_id="{{batch}}".format(batch="c" * 40)),
        ]


def initialise(self, token=None):
    self._token = True if token is None else token
    self._api = Api()
    self._filesystem = None


HfApiAdapter.__init__ = initialise


def progress(**kwargs):
    batch = finalisation._Batch(
        "batch-001", "c" * 40, 1, 1, {{}},
        (finalisation._Shard("data/train/part-00000.parquet", 1, 1, "d" * 64),),
    )
    finalisation._check_batch_ancestry(
        kwargs["hub"], kwargs["repository"], kwargs["revision"], (batch,)
    )
    return {{"pass": True}}


finalisation.finalise_p1_corpus = progress
""",
    )

    result = _run_cli(
        _FINALISER,
        "--revision",
        _REVISION,
        "--run-root",
        str(tmp_path / "run"),
        "--pipeline-config-sha256",
        _DIGEST,
        "--report",
        str(tmp_path / "report.json"),
        extra_python_path=tmp_path,
    )

    assert result.returncode == 0
    assert result.stdout == "P1 finalisation passed\n"
    assert result.stderr == ""


def test_finaliser_success_emits_only_status(tmp_path: Path) -> None:
    """A successful finaliser emits no report data or diagnostics."""
    _write_sitecustomize(
        tmp_path,
        """
import builtins

_original_import = builtins.__import__


def _import(name, globals=None, locals=None, fromlist=(), level=0):
    module = _original_import(name, globals, locals, fromlist, level)
    if name == "p1_dataset.finalisation":
        module.finalise_p1_corpus = lambda **kwargs: {"pass": True}
    return module


builtins.__import__ = _import
""",
    )

    result = _run_cli(
        _FINALISER,
        "--revision",
        _REVISION,
        "--run-root",
        str(tmp_path / "run"),
        "--pipeline-config-sha256",
        _DIGEST,
        "--report",
        str(tmp_path / "report.json"),
        extra_python_path=tmp_path,
    )

    assert result.returncode == 0
    assert result.stdout == "P1 finalisation passed\n"
    assert result.stderr == ""


def test_sanity_gate_help_has_no_import_warning_or_absolute_path() -> None:
    """Sanity-gate help is successful and does not expose its checkout path."""
    result = _run_cli(_SANITY_GATE, "--help")

    assert result.returncode == 0
    output = result.stdout + result.stderr
    assert "usage:" in result.stdout
    assert 'Field name "schema"' not in output
    assert str(_PROJECT_ROOT) not in output


def test_sanity_gate_hub_failure_is_sanitised(tmp_path: Path) -> None:
    """A Hub failure cannot leak candidate or request content."""
    _write_sitecustomize(
        tmp_path,
        """
import builtins

_original_import = builtins.__import__


def _import(name, globals=None, locals=None, fromlist=(), level=0):
    module = _original_import(name, globals, locals, fromlist, level)
    if name == "p1_dataset.publish":
        def fail_repo_info(self, *args, **kwargs):
            raise RuntimeError(
                "RAW_HUB_ERROR https://evil.example/resolve/commit?token=credential-secret"
            )
        module.HfApiAdapter.repo_info = fail_repo_info
    return module


builtins.__import__ = _import
""",
    )
    candidates = []
    for index in range(12):
        candidates.append(
            {
                "status": "accepted",
                "repository": "syvai/p1-segments",
                "revision": _REVISION,
                "parquet_path": "data/train/part-00.parquet",
                "row_index": index,
                "segment_id": f"segment-secret-{index}",
                "metadata_sha256": "c" * 64,
                "audio_sha256": "d" * 64,
                "payload": "payload-secret",
                "signed_url": (
                    "https://evil.example/resolve/commit?token=credential-secret"
                ),
            }
        )
    input_path = tmp_path / "sensitive-candidates.jsonl"
    input_path.write_text(
        "".join(json.dumps(candidate) + "\n" for candidate in candidates),
        encoding="utf-8",
    )

    result = _run_cli(
        _SANITY_GATE,
        "--input",
        str(input_path),
        "--report",
        str(tmp_path / "report.json"),
        "--pilot-head",
        _REVISION,
        "--pipeline-config-sha256",
        _DIGEST,
        extra_python_path=tmp_path,
    )

    _assert_safe_failure(
        result,
        "RAW_HUB_ERROR",
        "segment-secret",
        "payload-secret",
        "https://evil.example/resolve/commit",
        "credential-secret",
    )
    assert "unable to verify the pinned dataset revision" in result.stderr


def test_sanity_gate_success_emits_only_diagnostic(tmp_path: Path) -> None:
    """A successful sanity gate emits only its fixed status diagnostic."""
    _write_sitecustomize(
        tmp_path,
        """
import builtins

_original_import = builtins.__import__


def _import(name, globals=None, locals=None, fromlist=(), level=0):
    module = _original_import(name, globals, locals, fromlist, level)
    if name == "p1_dataset.v8_sanity_gate":
        module.run_v8_sanity_gate = lambda *args, **kwargs: {"pass": True}
    return module


builtins.__import__ = _import
""",
    )
    input_path = tmp_path / "candidates.jsonl"
    input_path.write_text("{}\n", encoding="utf-8")

    result = _run_cli(
        _SANITY_GATE,
        "--input",
        str(input_path),
        "--report",
        str(tmp_path / "report.json"),
        "--pilot-head",
        _REVISION,
        "--pipeline-config-sha256",
        _DIGEST,
        extra_python_path=tmp_path,
    )

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == "INFO: P1 v8 sanity gate passed\n"
