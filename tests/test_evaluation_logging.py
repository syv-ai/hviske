"""Regression tests for evaluation logging policy."""

import logging
from pathlib import Path

from hydra import compose
from hydra.core.utils import configure_log
from omegaconf import DictConfig

_TRANSPORT_LOGGERS = ("httpx", "httpcore", "huggingface_hub", "fsspec")


def test_custom_hydra_logging_suppresses_transport_info(tmp_path: Path) -> None:
    """The custom Hydra policy suppresses INFO from transport loggers."""
    config: DictConfig = compose(config_name="evaluation", return_hydra_config=True)
    config.hydra.runtime.output_dir = str(tmp_path)
    config.hydra.job.name = "evaluation"

    root_logger = logging.getLogger()
    previous_root_level = root_logger.level
    previous_root_handlers = list(root_logger.handlers)
    transport_loggers = tuple(logging.getLogger(name) for name in _TRANSPORT_LOGGERS)
    previous_transport_state = {
        logger: (logger.level, list(logger.handlers), logger.propagate, logger.disabled)
        for logger in transport_loggers
    }
    records: list[logging.LogRecord] = []
    collector = _RecordCollector(records)

    try:
        # Keep pytest's handlers out of dictConfig so the test does not close them.
        root_logger.handlers[:] = []
        configure_log(config.hydra.job_logging)
        root_logger.addHandler(collector)
        for logger in transport_loggers:
            logger.info("temporary signed dataset URL")

        assert records == []
        assert all(
            not logger.isEnabledFor(logging.INFO) for logger in transport_loggers
        )
    finally:
        root_logger.removeHandler(collector)
        for handler in root_logger.handlers:
            if handler not in previous_root_handlers:
                handler.close()
        root_logger.handlers[:] = previous_root_handlers
        root_logger.setLevel(previous_root_level)
        for logger, state in previous_transport_state.items():
            level, handlers, propagate, disabled = state
            logger.setLevel(level)
            logger.handlers[:] = handlers
            logger.propagate = propagate
            logger.disabled = disabled


class _RecordCollector(logging.Handler):
    """Collect records without writing them to a stream."""

    def __init__(self, records: list[logging.LogRecord]) -> None:
        super().__init__()
        self.records = records

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def test_evaluation_uses_custom_hydra_logging() -> None:
    """Evaluation composes the project's custom Hydra job logging policy."""
    config = compose(config_name="evaluation", return_hydra_config=True)

    assert config.hydra.runtime.choices["hydra/job_logging"] == "custom"
    assert set(config.hydra.job_logging.loggers) == set(_TRANSPORT_LOGGERS)
    assert all(
        config.hydra.job_logging.loggers[name].level == "WARNING"
        for name in _TRANSPORT_LOGGERS
    )
