"""MLFlow experiment tracking setup class."""

import mlflow

from .extracking_setup import ExTrackingSetup


class MLFlowSetup(ExTrackingSetup):
    """MLFlow setup class."""

    def run_finalization(self, exit_code: int = 0) -> None:
        """Finish the MLflow run and record whether training succeeded.

        Args:
            exit_code (optional):
                The process exit code to report. Defaults to ``0``.
        """
        status = "FINISHED" if exit_code == 0 else "FAILED"
        try:
            mlflow.end_run(status=status)
        except TypeError:
            # Older MLflow releases do not accept the status keyword.
            mlflow.end_run()

    def run_initialization(self) -> None:
        """Run the initialization of the experiment tracking setup."""
        mlflow.set_experiment(self.config.experiment_tracking.name_experiment)
        mlflow.start_run(run_name=self.config.experiment_tracking.name_run)
        return
