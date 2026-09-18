"""Regression tests for the executable Sparkie training runbook."""

from pathlib import Path

RUNBOOK = Path(__file__).parents[1] / "SPARKIE.md"
_REQUIRED_UNSETS = (
    "WANDB_API_KEY",
    "WANDB_ENTITY",
    "WANDB_BASE_URL",
    "WANDB_RUN_ID",
    "WANDB_RESUME",
    "WANDB_NAME",
    "WANDB_RUN_GROUP",
)


def test_sparkie_runbook_documents_p1_gate_and_publication() -> None:
    """The runbook retains the consumed P1 gate and publication command."""
    runbook = RUNBOOK.read_text()

    assert "P1_SEGMENTS_REVISION" not in runbook
    assert "syvai/p1-segments" in runbook
    assert "archives/hviske-p1-pipeline/f3dcf16/" in runbook
    assert "44284e5849b6b1d96b874891c579654a644e0e2f" in runbook
    assert "manually gated" in runbook
    assert "push_to_hub=true" in runbook
    assert "private=true private_only=true" in runbook


def test_sparkie_tmux_launches_scrub_inherited_wandb_identity() -> None:
    """The documented training process starts with a clean W&B identity."""
    runbook = RUNBOOK.read_text()
    launch_commands = [
        line for line in runbook.splitlines() if line.startswith('  "env ')
    ]

    assert len(launch_commands) == 1
    command = launch_commands[0]
    assert all(f"-u {variable}" in command for variable in _REQUIRED_UNSETS)
    assert "WANDB_PROJECT=$WANDB_PROJECT" in command
    assert "WANDB_MODE=online" in command
    assert "WANDB_LOG_MODEL=false" in command
    assert "WANDB_WATCH=false" in command
