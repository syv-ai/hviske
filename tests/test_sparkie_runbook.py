"""Regression tests for executable Sparkie training commands."""

from pathlib import Path

RUNBOOK = Path(__file__).parents[1] / "SPARKIE.md"
TRAINING_PLAN = Path(__file__).parents[1] / "docs" / "initial-training-plan.md"
_REQUIRED_UNSETS = (
    "WANDB_API_KEY",
    "WANDB_ENTITY",
    "WANDB_BASE_URL",
    "WANDB_RUN_ID",
    "WANDB_RESUME",
    "WANDB_NAME",
    "WANDB_RUN_GROUP",
)


def test_sparkie_full_commands_match_training_plan_evaluation_cadence() -> None:
    """Full-run commands retain the documented 2,000-step evaluation cadence."""
    runbook = RUNBOOK.read_text()
    training_plan = TRAINING_PLAN.read_text()
    full_commands = [
        line
        for line in runbook.splitlines()
        if line.startswith('  "env ')
        and "name_run=v6.0-tdt-full-200k-max8-seed-4242" in line
    ]

    assert len(full_commands) == 2
    assert all("--config-name bilingual" in command for command in full_commands)
    assert all("model_id=hviske-v6.0" in command for command in full_commands)
    assert all("model_dir=$full_dir_q" in command for command in full_commands)
    assert all("eval_steps=" not in command for command in full_commands)
    assert all("max_steps=" not in command for command in full_commands)
    assert (
        "Evaluate the full frozen development suite every 2,000 steps" in training_plan
    )
    assert "src/scripts/publish_model.py" in runbook
    assert "runs/hviske-v6.0 syvai/hviske-v6.0 --private" in runbook


def test_sparkie_runbook_documents_materialised_worker_safety() -> None:
    """The runbook explains the local artefact and worker-count contract."""
    runbook = RUNBOOK.read_text()

    assert "dataset_num_workers=1" in runbook
    assert "dataloader_num_workers=2" in runbook
    assert "HVISKE_MATERIALISED_OVERLAYS_ROOT" in runbook
    assert "package-level materialised-overlay implementation" in runbook
    assert "CLOSE-WAIT" in runbook
    assert "not a one-worker child process" in runbook
    assert "two spawned DataLoader workers" in runbook
    assert "85–86°C" in runbook
    assert "positional equality checks strict" in runbook
    assert "do not relax them" in runbook
    assert "fails closed" in runbook


def test_sparkie_tmux_launches_scrub_inherited_wandb_identity() -> None:
    """Every tmux training process starts with a clean W&B identity."""
    runbook = RUNBOOK.read_text()
    launch_commands = [
        line for line in runbook.splitlines() if line.startswith('  "env ')
    ]

    assert len(launch_commands) == 6
    for command in launch_commands:
        assert all(f"-u {variable}" in command for variable in _REQUIRED_UNSETS)
        assert "WANDB_PROJECT=$wandb_project_q" in command
        assert "WANDB_MODE=online" in command
        assert "WANDB_LOG_MODEL=false" in command
        assert "WANDB_WATCH=false" in command
