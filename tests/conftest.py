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

    # Ensure Gyan FFmpeg and conda env's libraries are on PATH if running on Windows
    conda_paths = [
        r"C:\Users\santo\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-essentials_build\bin",
        r"C:\Users\santo\miniconda3\envs\incubrix",
        r"C:\Users\santo\miniconda3\envs\incubrix\Scripts",
        r"C:\Users\santo\miniconda3\envs\incubrix\Library\bin",
        r"C:\Users\santo\miniconda3\envs\incubrix\Library\usr\bin",
        r"C:\Users\santo\miniconda3\envs\incubrix\Library\mingw-w64\bin",
        r"C:\Users\santo\miniconda3\envs\incubrix\bin",
    ]
    cur_path = os.environ.get("PATH", "")
    new_parts = [p for p in conda_paths if Path(p).exists() and p not in cur_path]
    if new_parts:
        os.environ["PATH"] = ";".join(new_parts) + ";" + cur_path

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
