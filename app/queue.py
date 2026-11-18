"""Pluggable background job queue supporting Redis (Docker worker) and bounded ThreadPoolExecutor (inline)."""

from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import redis

from app.config import Settings, get_settings
from app.logging_config import get_logger
from app.pipeline.orchestrator import execute_broll_job

logger = get_logger("queue")
REDIS_QUEUE_KEY = "incubrix:broll:jobs"


class JobQueue:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.backend = self.settings.queue_backend
        self._executor: Optional[ThreadPoolExecutor] = None
        self._redis: Optional[redis.Redis] = None

        if self.backend == "redis" and self.settings.redis_url:
            self._redis = redis.Redis.from_url(self.settings.redis_url, decode_responses=True)
        else:
            self.backend = "inline"
            self._executor = ThreadPoolExecutor(
                max_workers=self.settings.worker_concurrency,
                thread_name_prefix="broll-worker",
            )

    def enqueue(self, job_id: str) -> None:
        if self.backend == "redis" and self._redis is not None:
            self._redis.rpush(REDIS_QUEUE_KEY, job_id)
            logger.info("job_enqueued_redis", job_id=job_id)
        else:
            assert self._executor is not None
            self._executor.submit(execute_broll_job, job_id)
            logger.info("job_enqueued_inline", job_id=job_id)

    def is_ready(self) -> bool:
        if self.backend == "redis" and self._redis is not None:
            try:
                return bool(self._redis.ping())
            except Exception:
                return False
        return self._executor is not None

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False)


_queue_instance: Optional[JobQueue] = None


def get_queue() -> JobQueue:
    global _queue_instance
    if _queue_instance is None:
        _queue_instance = JobQueue()
    return _queue_instance


def reset_queue() -> None:
    global _queue_instance
    if _queue_instance is not None:
        _queue_instance.shutdown()
    _queue_instance = None
