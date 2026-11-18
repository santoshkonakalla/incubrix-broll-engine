"""Dedicated Redis background worker container entrypoint (`python -m app.worker`)."""

import signal
import sys
import time

import redis

from app.config import get_settings
from app.db import init_db
from app.logging_config import configure_logging, get_logger
from app.pipeline.orchestrator import execute_broll_job
from app.queue import REDIS_QUEUE_KEY

logger = get_logger("worker")
_running = True


def _handle_sig(_signum, _frame) -> None:
    global _running
    _running = False
    logger.info("worker_shutdown_signal_received")


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    init_db(settings.database_url)

    if not settings.redis_url:
        logger.error("worker_missing_redis_url")
        sys.exit(1)

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
    logger.info("worker_started", redis_url=settings.redis_url, adapter_mode=settings.adapter_mode)

    while _running:
        try:
            item = client.blpop(REDIS_QUEUE_KEY, timeout=2)
            if not item:
                continue
            _, job_id = item
            logger.info("worker_picked_job", job_id=job_id)
            execute_broll_job(job_id)
        except redis.RedisError as exc:
            logger.warning("worker_redis_error", error=str(exc))
            time.sleep(2)
        except Exception as exc:
            logger.error("worker_unexpected_error", error=str(exc))
            time.sleep(1)


if __name__ == "__main__":
    main()
