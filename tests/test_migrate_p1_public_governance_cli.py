"""Subprocess tests for privacy-safe P1 migration CLI output."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parents[1]
_SCRIPT = _PROJECT_ROOT / "src/scripts/migrate_p1_public_governance.py"
_SENTINEL = "SENTINEL_PATH https://evil.example/?token=SENTINEL_CONTENT"


def test_migration_cli_captures_import_warning_logs_and_hub_content(
    tmp_path: Path,
) -> None:
    """Import warnings, dependency logs, and Hub content never reach the CLI."""
    (tmp_path / "sitecustomize.py").write_text(
        """
import builtins
import logging
import warnings

_original_import = builtins.__import__


def _import(name, globals=None, locals=None, fromlist=(), level=0):
    module = _original_import(name, globals, locals, fromlist, level)
    if name == "p1_dataset.governance":
        warnings.warn("SENTINEL_PATH?token=SENTINEL_CONTENT", UserWarning)
        logging.getLogger("httpx").warning("SENTINEL_URL response SENTINEL_CONTENT")
        from huggingface_hub.errors import HfHubHTTPError

        def fail(**kwargs):
            del kwargs
            response_type = type(
                "Response",
                (),
                {
                    "status_code": 503,
                    "request": type(
                        "Request", (), {"url": "https://evil/?token=SENTINEL"}
                    )(),
                    "headers": {"X-Error-Message": "SENTINEL_CONTENT"},
                    "text": "SENTINEL_CONTENT",
                },
            )
            raise HfHubHTTPError("SENTINEL_CONTENT", response=response_type())

        module.migrate_public_governance = fail
    return module


builtins.__import__ = _import
""",
        encoding="utf-8",
    )

    result = _run_cli("--expected-old-head", "a" * 40, extra_python_path=tmp_path)

    assert result.returncode == 1
    output = result.stdout + result.stderr
    assert output == (
        "P1 governance migration failed class=HfHubHTTPError "
        "status=503 phase=unknown reason=server_error\n"
    )
    assert "SENTINEL" not in output


def _run_cli(
    *arguments: str, extra_python_path: Path | None = None
) -> subprocess.CompletedProcess[str]:
    """Run the migration CLI with a controlled import path.

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
        [sys.executable, str(_SCRIPT), *arguments],
        cwd=_PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_migration_cli_help_has_no_absolute_path() -> None:
    """Help output is fixed and does not expose the checkout path."""
    result = _run_cli("--help")

    assert result.returncode == 0
    output = result.stdout + result.stderr
    assert str(_PROJECT_ROOT) not in output
    assert output == (
        "Usage: migrate-p1-public-governance --expected-old-head SHA "
        "[--apply] [--config FILE]\n\n"
        "Validate or apply the P1 public-governance migration.\n\n"
        "Options:\n"
        "  --expected-old-head SHA  Expected previous revision.\n"
        "  --apply                   Apply the migration.\n"
        "  --config FILE             Configuration file.\n"
        "  -h, --help                Show this help.\n"
    )


def test_migration_cli_rejects_malformed_arguments_without_echoing_input() -> None:
    """Malformed arguments produce only the fixed argument diagnostic."""
    result = _run_cli("--unexpected", _SENTINEL)

    assert result.returncode == 2
    assert result.stdout + result.stderr == (
        "P1 governance migration failed category=arguments\n"
    )
    assert _SENTINEL not in result.stdout + result.stderr
