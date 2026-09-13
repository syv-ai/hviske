"""Safely migrate existing P1 metadata from private to public governance."""

from __future__ import annotations

import argparse
import collections.abc as c
import contextlib
import logging
import os
import re
import sys
import typing as t
import warnings
from pathlib import Path

_USAGE = (
    "Usage: migrate-p1-public-governance --expected-old-head SHA "
    "[--apply] [--config FILE]"
)
_SAFE_PHASES = {
    "unknown",
    "preupload",
    "lfs_batch",
    "xet",
    "commit",
    "paths_info",
    "repo_tree",
    "repo_create",
}
_SAFE_REASONS = {
    "unknown",
    "revision_not_found",
    "entry_not_found",
    "repository_not_found",
    "authorisation",
    "missing_uploaded_object",
    "stale_parent",
    "xet_unavailable",
    "rate_limited",
    "server_error",
}


def main(argv: c.Sequence[str] | None = None) -> int:
    """Validate and optionally apply the one-shot governance migration.

    Args:
        argv (optional):
            Arguments to parse, or ``None`` to use the process arguments.

    Returns:
        Zero on success, otherwise a privacy-safe diagnostic status.
    """
    try:
        _configure_output_isolation()
        try:
            arguments = _parse_arguments(argv)
        except _HelpRequested:
            _emit(_help_text())
            return 0
        except (argparse.ArgumentError, _ArgumentFailure):
            _emit("P1 governance migration failed category=arguments", error=True)
            return 2

        report = _perform_migration(arguments)
        _emit_success(report)
        return 0
    except KeyboardInterrupt:
        _emit("P1 governance migration interrupted", error=True)
        return 130
    except BaseException as error:
        _emit_failure(error)
        return 1


def _configure_output_isolation() -> None:
    """Disable process-wide logging before importing project dependencies."""
    logging.disable(logging.CRITICAL + 1)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(logging.NullHandler())
    root.setLevel(logging.CRITICAL + 1)
    for value in logging.root.manager.loggerDict.values():
        if isinstance(value, logging.Logger):
            value.handlers.clear()
            value.disabled = True


def _emit(message: str, *, error: bool = False) -> None:
    """Write one fixed-format line, ignoring broken output pipes safely."""
    stream = sys.stderr if error else sys.stdout
    try:
        stream.write(f"{message}\n")
        stream.flush()
    except BaseException:
        return


def _emit_failure(error: BaseException) -> None:
    """Emit only a bounded diagnostic for a migration failure."""
    try:
        diagnostic = _hub_diagnostic(error)
    except BaseException:
        diagnostic = None
    if diagnostic is None:
        _emit(
            "P1 governance migration failed "
            f"class={_safe_class_name(type(error).__name__)}",
            error=True,
        )
        return

    status = diagnostic.status_code
    safe_status = (
        str(status) if isinstance(status, int) and 100 <= status <= 599 else "none"
    )
    phase = _safe_member(getattr(diagnostic, "phase", None), _SAFE_PHASES)
    reason = _safe_member(getattr(diagnostic, "reason", None), _SAFE_REASONS)
    _emit(
        "P1 governance migration failed "
        f"class={_safe_class_name(type(error).__name__)} status={safe_status} "
        f"phase={phase} reason={reason}",
        error=True,
    )


def _hub_diagnostic(error: BaseException) -> object | None:
    """Classify genuine Hub errors while suppressing classifier side effects.

    Returns:
        A bounded Hub diagnostic, or ``None`` for another exception type.
    """
    with _quiet_runtime():
        from huggingface_hub.errors import HfHubHTTPError

        from p1_dataset.hub_diagnostics import classify_hub_error

        if not isinstance(error, HfHubHTTPError):
            return None
        return classify_hub_error(error)


@contextlib.contextmanager
def _quiet_runtime() -> c.Iterator[None]:
    """Suppress warnings and dependency streams during untrusted operations."""
    with (
        warnings.catch_warnings(),
        open(os.devnull, "w", encoding="utf-8") as sink,
        contextlib.redirect_stdout(sink),
        contextlib.redirect_stderr(sink),
    ):
        warnings.simplefilter("ignore")
        yield


def _safe_class_name(value: str) -> str:
    """Keep exception class names within a fixed, non-sensitive alphabet.

    Returns:
        The class name when it is safe, otherwise ``unknown``.
    """
    if re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", value):
        return value
    return "unknown"


def _safe_member(value: object, allowed: set[str]) -> str:
    """Return an allowlisted diagnostic field or ``unknown``."""
    return value if isinstance(value, str) and value in allowed else "unknown"


def _emit_success(report: object) -> None:
    """Emit bounded aggregate success output without repository evidence."""
    applied = getattr(report, "applied", None) is True
    file_count = getattr(report, "file_count", None)
    if not isinstance(file_count, int) or isinstance(file_count, bool):
        file_count = 0
    file_count = max(0, min(file_count, 999_999_999))
    mode = "applied" if applied else "validated"
    _emit(f"P1 governance migration succeeded mode={mode} files={file_count}")


def _help_text() -> str:
    """Return help text that contains no process or repository paths."""
    return (
        f"{_USAGE}\n\n"
        "Validate or apply the P1 public-governance migration.\n\n"
        "Options:\n"
        "  --expected-old-head SHA  Expected previous revision.\n"
        "  --apply                   Apply the migration.\n"
        "  --config FILE             Configuration file.\n"
        "  -h, --help                Show this help."
    )


def _parse_arguments(argv: c.Sequence[str] | None) -> argparse.Namespace:
    """Parse arguments without allowing input-controlled diagnostics.

    Returns:
        Parsed arguments.
    """
    parser = _SafeArgumentParser(
        prog="migrate-p1-public-governance",
        usage=_USAGE,
        description="Validate or apply the P1 public-governance migration.",
        add_help=True,
        allow_abbrev=False,
        exit_on_error=False,
    )
    parser.add_argument("--expected-old-head", required=True, metavar="SHA")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--config", type=Path, default=Path("config/p1_segments.yaml"), metavar="FILE"
    )
    return parser.parse_args(argv)


class _SafeArgumentParser(argparse.ArgumentParser):
    """Argument parser that never exposes argparse's input-controlled text."""

    def error(self, message: str) -> t.Never:
        """Reject malformed arguments without echoing the parser diagnostic.

        Raises:
            _ArgumentFailure:
                Always, so the caller can emit a fixed diagnostic.
        """
        del message
        raise _ArgumentFailure

    def exit(self, status: int = 0, message: str | None = None) -> t.Never:
        """Handle help and errors without writing argparse output.

        Raises:
            _HelpRequested:
                When argparse requests successful help exit.
            _ArgumentFailure:
                For all other exits.
        """
        del message
        if status == 0:
            raise _HelpRequested
        raise _ArgumentFailure

    def print_help(self, file: object | None = None) -> None:
        """Discard argparse help before the fixed help text is emitted."""
        del file


def _perform_migration(arguments: argparse.Namespace) -> object:
    """Import migration dependencies and run them with output capture enabled.

    Returns:
        The aggregate migration report.
    """
    with _quiet_runtime():
        from omegaconf import DictConfig, OmegaConf

        from p1_dataset.governance import migrate_public_governance
        from p1_dataset.pipeline import PipelineSettings
        from p1_dataset.publish import HfApiAdapter

        settings = PipelineSettings.from_config(
            t.cast(DictConfig, OmegaConf.load(arguments.config))
        )
        return migrate_public_governance(
            api=HfApiAdapter(),
            settings=settings,
            expected_old_head=arguments.expected_old_head,
            apply=arguments.apply,
        )


class _ArgumentFailure(Exception):
    """Signal that arguments did not pass the sanitised boundary."""


class _HelpRequested(Exception):
    """Signal that the fixed help text should be emitted."""


if __name__ == "__main__":
    raise SystemExit(main())
