"""Pytest fixtures: completely isolated zero-API-key mock environment per test session."""

import os
from pathlib import Path
import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def test_env(tmp_path_factory):
    base = tmp_path_factory.mktemp("incubrix_test")
    env_vars = {
        "ADAPTER_MODE": "mock",
        "ASR_PROVIDER": "mock",
        "TTS_PROVIDER": "mock",
        "PLANNER_PROVIDER": "mock",
        "VLM_SCORER_PROVIDER": "mock",
        "FOOTAGE_SOURCES": "pexels,pixabay,generative,statcard",
        "DATABASE_URL": f"sqlite:///{(base / 'test.db').as_posix()}",
        "QUEUE_BACKEND": "inline",
        "STORAGE_BACKEND": "local",
        "LOCAL_STORAGE_DIR": str(base / "storage"),
        "TEMP_WORK_DIR": str(base / "tmp"),
        "RATE_LIMIT_PER_MINUTE": "200",
        "API_ACCESS_TOKEN": "",
        "WEBHOOK_HMAC_SECRET": "test-hmac-secret",
    }
    for k, v in env_vars.items():
        os.environ[k] = v

    # Ensure conda env's ffmpeg is on PATH if running from Windows without activation
    conda_bin = Path(r"C:\Users\santo\miniconda3\envs\incubrix\Library\bin")
    if conda_bin.exists() and str(conda_bin) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{conda_bin};{os.environ.get('PATH', '')}"

    from app.config import get_settings
    from app.db import init_db
    from app.queue import reset_queue
    from app.storage import reset_storage

    get_settings.cache_clear()
    reset_storage()
    reset_queue()
    init_db(env_vars["DATABASE_URL"])
    return base


@pytest.fixture(scope="session")
def client(test_env):
    from app.main import create_app

    app = create_app()
    with TestClient(app) as tc:
        yield tc


def poll_until_done(client: TestClient, job_id: str, timeout: float = 90.0) -> dict:
    deadline = time.perf_counter() + timeout
    last = {}
    while time.perf_counter() < deadline:
        resp = client.get(f"/api/v1/jobs/{job_id}")
        assert resp.status_code == 200
        last = resp.json()
        if last["status"] in ("succeeded", "failed"):
            return last
        time.sleep(0.25)
    raise AssertionError(f"Job {job_id} did not finish within {timeout}s; last={last}")
