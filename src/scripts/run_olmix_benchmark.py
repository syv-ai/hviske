"""Run the serial Olmix anchor calibration matrix."""

from __future__ import annotations

import argparse
import ast
import collections.abc as c
import json
import logging
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

MODELS = ("whisper-xxsmall", "hviske-v5-tiny")
ANCHORS = ("olmix_baseline", "olmix_read_speech_heavy", "olmix_spontaneous_heavy")
EVALUATION_STEPS = (250, 500, 1000, 2000, 3000)
MODEL_REVISIONS = {
    "whisper-xxsmall": "169d4a4341b33bc18d8881c4b69c2e104e1cc0af",
    "hviske-v5-tiny": "361051e8ed732798d68fcd5d5ec64fd4e39da40b",
}
FINETUNE_SCRIPT = "src/scripts/finetune_asr_model.py"
LOGGER = logging.getLogger("olmix_benchmark")


def main() -> int:
    """Parse options and run the requested calibration jobs.

    Returns:
        Zero after every requested job completes successfully.
    """
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.matrix:
        jobs = [(model, anchor) for model in MODELS for anchor in ANCHORS]
        if not args.skip_smoke:
            LOGGER.info("Running the two-model, two-step smoke before the matrix")
            for model in MODELS:
                _run_job(
                    model=model,
                    anchor="olmix_baseline",
                    output_root=output_root,
                    max_steps=2,
                    evaluation_steps=(2,),
                    run_kind="smoke",
                )
    elif args.smoke:
        jobs = [(model, "olmix_baseline") for model in MODELS]
    else:
        jobs = [(args.model, args.anchor)]

    for model, anchor in jobs:
        _run_job(
            model=model,
            anchor=anchor,
            output_root=output_root,
            max_steps=2 if args.smoke else 3000,
            evaluation_steps=(2,) if args.smoke else EVALUATION_STEPS,
            run_kind="smoke" if args.smoke else "matrix",
        )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--matrix", action="store_true", help="Run all six jobs serially."
    )
    mode.add_argument(
        "--smoke",
        action="store_true",
        help="Run the two-step baseline smoke for both model configurations.",
    )
    mode.add_argument(
        "--model", choices=MODELS, help="Model config for one selected job."
    )
    parser.add_argument(
        "--anchor", choices=ANCHORS, help="Anchor for one selected job."
    )
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Do not run the two-model smoke before --matrix.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/olmix"),
        help="Directory for per-job metadata, logs, and model outputs.",
    )
    args = parser.parse_args()
    if args.model is not None and args.anchor is None:
        parser.error("--model requires --anchor")
    if args.anchor is not None and args.model is None:
        parser.error("--anchor requires --model")
    if args.skip_smoke and not args.matrix:
        parser.error("--skip-smoke is only valid with --matrix")
    return args


def _run_job(
    *,
    model: str,
    anchor: str,
    output_root: Path,
    max_steps: int,
    evaluation_steps: tuple[int, ...],
    run_kind: str,
) -> None:
    run_id = _make_run_id(model=model, anchor=anchor, run_kind=run_kind)
    run_dir = output_root / run_id
    model_id = f"olmix-{model}-{anchor}-{run_id.rsplit('-', 1)[-1]}"
    models_dir = run_dir / "models"
    run_dir.mkdir(parents=True, exist_ok=False)
    command = build_command(
        model=model,
        anchor=anchor,
        model_id=model_id,
        models_dir=models_dir,
        hydra_run_dir=run_dir / "hydra",
        max_steps=max_steps,
        evaluation_steps=evaluation_steps,
        metrics_path=run_dir / "evaluation_metrics.jsonl",
    )
    metadata: dict[str, object] = {
        "model": model,
        "anchor": anchor,
        "run_kind": run_kind,
        "run_id": run_id,
        "model_id": model_id,
        "run_dir": str(run_dir),
        "model_dir": str(models_dir / model_id),
        "evaluation_metrics_path": str(run_dir / "evaluation_metrics.jsonl"),
        "evaluation_steps": list(evaluation_steps),
        "model_revision": MODEL_REVISIONS[model],
        "max_steps": max_steps,
        "validation_cap": 500,
        "commit": _git_commit(),
        "command": command,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "metrics": [],
    }
    metadata_path = run_dir / "metadata.json"
    _write_metadata(path=metadata_path, metadata=metadata)

    LOGGER.info("Starting %s / %s in %s", model, anchor, run_dir)
    started = time.monotonic()
    metrics: list[dict[str, object]] = []
    try:
        with (run_dir / "run.log").open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                cwd=Path.cwd(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log_file.write(line)
                log_file.flush()
                LOGGER.info("[%s] %s", run_id, line.rstrip())
                metric = _extract_metrics(line)
                if metric is not None:
                    metrics.append(metric)
            return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)
    except BaseException:
        metadata.update(
            {
                "status": "failed",
                "duration_seconds": time.monotonic() - started,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "metrics": _read_metrics(
                    path=run_dir / "evaluation_metrics.jsonl", fallback=metrics
                ),
            }
        )
        _write_metadata(path=metadata_path, metadata=metadata)
        raise
    metadata.update(
        {
            "status": "completed",
            "duration_seconds": time.monotonic() - started,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "metrics": _read_metrics(
                path=run_dir / "evaluation_metrics.jsonl", fallback=metrics
            ),
        }
    )
    _write_metadata(path=metadata_path, metadata=metadata)


def _extract_metrics(line: str) -> dict[str, object] | None:
    start, end = line.find("{"), line.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = ast.literal_eval(line[start : end + 1])
    except (SyntaxError, ValueError):
        return None
    if not isinstance(parsed, c.Mapping):
        return None
    metric: dict[str, object] = {}
    for key, value in parsed.items():
        if not isinstance(key, str):
            continue
        if not (key.startswith("eval_") or key in {"step", "loss"}):
            continue
        if isinstance(value, (bool, float, int, str)) or value is None:
            metric[key] = value
    return metric or None


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _make_run_id(*, model: str, anchor: str, run_kind: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{run_kind}-{model}-{anchor}-{timestamp}-{uuid.uuid4().hex[:8]}"


def _read_metrics(
    *, path: Path, fallback: list[dict[str, object]]
) -> list[dict[str, object]]:
    if not path.exists():
        return fallback
    metrics: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and isinstance(record.get("step"), int):
            metrics.append(record)
    return metrics


def _write_metadata(*, path: Path, metadata: dict[str, object]) -> None:
    path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_command(
    *,
    model: str,
    anchor: str,
    model_id: str,
    models_dir: Path,
    hydra_run_dir: Path,
    max_steps: int = 3000,
    evaluation_steps: tuple[int, ...] = EVALUATION_STEPS,
    metrics_path: Path | None = None,
) -> list[str]:
    """Build a credential-free finetuning command for one calibration job.

    The Trainer accepts one integer ``eval_steps`` value.  The launcher therefore
    sets it to one and passes the desired non-uniform schedule to the callback-aware
    config key ``evaluation_steps``; the callback suppresses all other evaluations.

    Returns:
        The complete subprocess command as argument tokens.
    """
    evaluation_override = "+evaluation_steps=[{}]".format(
        ",".join(str(step) for step in evaluation_steps)
    )
    return [
        "uv",
        "run",
        "python",
        FINETUNE_SCRIPT,
        "--config-name",
        "sparkie_bilingual",
        f"+anchors={anchor}",
        f"model={model}",
        f"model_id={model_id}",
        f"models_dir={models_dir.resolve()}",
        f"hydra.run.dir={hydra_run_dir.resolve()}",
        f"max_steps={max_steps}",
        evaluation_override,
        *(
            [f"+evaluation_metrics_path={metrics_path.resolve()}"]
            if metrics_path is not None
            else []
        ),
        "eval_steps=1",
        f"save_steps={max_steps}",
        "save_total_limit=1",
        "max_validation_samples_per_dataset=500",
        "push_to_hub=false",
        "create_pr=false",
        "enable_experiment_tracking=false",
        "private=true",
        "private_only=true",
    ]


if __name__ == "__main__":
    sys.exit(main())
