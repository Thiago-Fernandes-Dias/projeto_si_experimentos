from contextlib import contextmanager
from datetime import datetime
import json
import logging
import os
import sys
from typing import Any, Iterator

from lib.utils import create_dir_if_not_exists


class _LoggingSettings:
    level: int = logging.INFO
    log_to_console: bool = True
    log_to_file: bool = True
    log_filename: str = "execution.log"
    results_base_dir: str = "results"


_SETTINGS = _LoggingSettings()


class JsonLinesFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects (.jsonl)."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        event = getattr(record, "event", None)
        if event is not None:
            payload["event"] = event

        structured_data = getattr(record, "structured_data", None)
        if isinstance(structured_data, dict):
            payload.update(structured_data)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, default=str)


class ConsoleStructuredFormatter(logging.Formatter):
    """Formats log records as readable console lines with key=value metadata."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        base = f"{timestamp} | {record.levelname:<5} | {record.name} | {record.getMessage()}"

        structured_data = getattr(record, "structured_data", None)
        if isinstance(structured_data, dict) and structured_data:
            kv_pairs = " ".join(f"{k}={v}" for k, v in structured_data.items())
            base = f"{base} | {kv_pairs}"

        if record.exc_info:
            base = f"{base}\n{self.formatException(record.exc_info)}"

        return base


def _resolve_level(level: int | str | None) -> int:
    if level is None:
        env_level = os.getenv("SI_LOG_LEVEL", "INFO").upper()
        return getattr(logging, env_level, logging.INFO)
    if isinstance(level, str):
        return getattr(logging, level.upper(), logging.INFO)
    return int(level)


def configure_logging(
    *,
    level: int | str | None = None,
    log_to_console: bool | None = None,
    log_to_file: bool | None = None,
    log_filename: str | None = None,
    results_base_dir: str | None = None,
) -> None:
    """Configures structured console logging and file logging settings for experiments."""
    _SETTINGS.level = _resolve_level(level)

    if log_to_console is not None:
        _SETTINGS.log_to_console = log_to_console
    else:
        _SETTINGS.log_to_console = os.getenv("SI_LOG_TO_CONSOLE", "1").lower() not in ("0", "false", "no")

    if log_to_file is not None:
        _SETTINGS.log_to_file = log_to_file
    else:
        _SETTINGS.log_to_file = os.getenv("SI_LOG_TO_FILE", "1").lower() not in ("0", "false", "no")

    if log_filename is not None:
        _SETTINGS.log_filename = log_filename
    else:
        _SETTINGS.log_filename = os.getenv("SI_LOG_FILENAME", _SETTINGS.log_filename)

    if results_base_dir is not None:
        _SETTINGS.results_base_dir = results_base_dir

    root_logger = logging.getLogger()
    root_logger.setLevel(_SETTINGS.level)

    # Replace existing stream handlers to avoid duplicate console lines
    for handler in list(root_logger.handlers):
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)

    if _SETTINGS.log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(_SETTINGS.level)
        console_handler.setFormatter(ConsoleStructuredFormatter())
        root_logger.addHandler(console_handler)


def _ensure_root_configured() -> None:
    root_logger = logging.getLogger()
    has_structured_console = any(
        isinstance(h.formatter, ConsoleStructuredFormatter) for h in root_logger.handlers
    )
    if not has_structured_console and _SETTINGS.log_to_console:
        configure_logging()


@contextmanager
def experiment_log_context(exp_name: str, start_time: datetime) -> Iterator[str | None]:
    """Attaches a JSONL file handler at results/<exp_name>/<timestamp>/<log_filename>."""
    _ensure_root_configured()
    if not _SETTINGS.log_to_file:
        yield None
        return

    run_dir = os.path.join(
        _SETTINGS.results_base_dir,
        exp_name,
        start_time.strftime("%Y-%m-%d_%H-%M-%S"),
    )
    create_dir_if_not_exists(run_dir)
    log_path = os.path.join(run_dir, _SETTINGS.log_filename)

    root_logger = logging.getLogger()
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(_SETTINGS.level)
    file_handler.setFormatter(JsonLinesFormatter())
    root_logger.addHandler(file_handler)
    try:
        yield log_path
    finally:
        file_handler.flush()
        root_logger.removeHandler(file_handler)
        file_handler.close()


def log_structured(
    logger: logging.Logger,
    level: int,
    message: str,
    *,
    event: str,
    **fields: Any,
) -> None:
    """Emits a log record enriched with an event name and structured key-value fields."""
    logger.log(
        level,
        message,
        extra={
            "event": event,
            "structured_data": fields,
        },
    )
