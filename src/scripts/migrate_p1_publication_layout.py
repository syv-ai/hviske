"""Offline migration of pending P1 Parquet publication paths."""

from __future__ import annotations

import argparse
import collections.abc as c
import contextlib
import logging
import os
import sys
import typing as t
from pathlib import Path

_USAGE = (
    "Usage: migrate-p1-publication-layout --run-root DIR "
    "--expected-digest SHA [--publication-lock FILE] [--apply]"
)


def main(argv: c.Sequence[str] | None = None) -> int:
    """Validate or apply the local-only publication layout migration.

    Returns:
        Zero for success, two for malformed arguments, or one for unsafe evidence.
    """
    _configure_output_isolation()
    try:
        arguments = _parse_arguments(argv)
        with _quiet_runtime():
            from p1_dataset.layout_migration import migrate_layout

            report = migrate_layout(
                run_root=arguments.run_root,
                expected_digest=arguments.expected_digest,
                publication_lock=arguments.publication_lock,
                apply=arguments.apply,
            )
        mode = "applied" if report.applied else "validated"
        _emit(
            f"P1 layout migration succeeded mode={mode} ledgers={report.ledger_count} "
            f"shards={report.shard_count} manifests={report.manifest_count} "
            f"backups={report.backup_count}"
        )
        return 0
    except _HelpRequested:
        _emit(_USAGE)
        return 0
    except _ArgumentFailure:
        _emit("P1 layout migration failed category=arguments", error=True)
        return 2
    except KeyboardInterrupt:
        _emit("P1 layout migration interrupted", error=True)
        return 130
    except BaseException:
        _emit("P1 layout migration failed category=validation", error=True)
        return 1


def _configure_output_isolation() -> None:
    logging.disable(logging.CRITICAL + 1)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(logging.NullHandler())
    root.setLevel(logging.CRITICAL + 1)


def _emit(message: str, *, error: bool = False) -> None:
    stream = sys.stderr if error else sys.stdout
    try:
        stream.write(f"{message}\n")
        stream.flush()
    except BaseException:
        pass


def _parse_arguments(argv: c.Sequence[str] | None) -> argparse.Namespace:
    raw_arguments = tuple(sys.argv[1:] if argv is None else argv)
    if "-h" in raw_arguments or "--help" in raw_arguments:
        raise _HelpRequested
    parser = _SafeArgumentParser(prog="migrate-p1-publication-layout", add_help=False)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--expected-digest", required=True)
    parser.add_argument("--publication-lock", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("-h", "--help", action="store_true")
    arguments = parser.parse_args(raw_arguments)
    if arguments.help:
        raise _HelpRequested
    return arguments


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> t.NoReturn:
        del message
        raise _ArgumentFailure


@contextlib.contextmanager
def _quiet_runtime() -> c.Iterator[None]:
    saved: dict[int, int] = {}
    null_fd: int | None = None
    try:
        for fd in (1, 2):
            saved[fd] = os.dup(fd)
        null_fd = os.open(os.devnull, os.O_WRONLY)
        for fd in (1, 2):
            os.dup2(null_fd, fd)
        yield
    finally:
        for fd, descriptor in saved.items():
            try:
                os.dup2(descriptor, fd)
                os.close(descriptor)
            except OSError:
                pass
        if null_fd is not None:
            try:
                os.close(null_fd)
            except OSError:
                pass


class _ArgumentFailure(Exception):
    pass


class _HelpRequested(Exception):
    pass


if __name__ == "__main__":
    raise SystemExit(main())
