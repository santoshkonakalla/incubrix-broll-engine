"""Structured JSON logging with timestamp, level, job_id, and correlation_id (Integration Standard #8)."""

import logging
import sys
from contextvars import ContextVar

import structlog

correlation_id_ctx: ContextVar[str] = ContextVar("correlation_id", default="-")
job_id_ctx: ContextVar[str] = ContextVar("job_id", default="-")


def _inject_context(_logger: object, _method_name: str, event_dict: dict) -> dict:
    event_dict.setdefault("correlation_id", correlation_id_ctx.get())
    event_dict.setdefault("job_id", job_id_ctx.get())
    return event_dict


def configure_logging(log_level: str = "INFO") -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level, force=True)

    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
            _inject_context,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "incubrix"):
    return structlog.get_logger(name)
