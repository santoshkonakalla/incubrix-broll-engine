"""Pytest fixtures: completely isolated zero-API-key mock environment per test session."""

import os
from pathlib import Path
import sys
import time

if sys.platform == "win32":
    _gyan_path = r"C:\Users\santo\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-essentials_build\bin"
    _cur_items = [p for p in os.environ.get("PATH", "").split(";") if p.strip()]
    _filtered = [p for p in _cur_items if "incubrix\\library\\bin" not in p.lower()]
    if Path(_gyan_path).exists():
        os.environ["PATH"] = _gyan_path + ";" + ";".join(_filtered)
    else:
        os.environ["PATH"] = ";".join(_filtered)

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

    # Ensure Gyan FFmpeg is on PATH on Windows without broken conda ffmpeg
    if sys.platform == "win32":
        gyan = r"C:\Users\santo\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-essentials_build\bin"
        cur_parts = [p for p in os.environ.get("PATH", "").split(";") if "incubrix\\library\\bin" not in p.lower() and p.strip()]
        if Path(gyan).exists():
            os.environ["PATH"] = gyan + ";" + ";".join(cur_parts)

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
