"""API integration tests verifying Integration Standard points 2, 3, 4, 5, 8."""

from pathlib import Path
import yaml

from tests.conftest import poll_until_done


def test_health_and_ready_endpoints(client):
    h = client.get("/health")
    assert h.status_code == 200
    assert h.json()["status"] == "ok"
    assert "X-Correlation-Id" in h.headers

    r = client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert body["checks"]["database"] is True
    assert body["checks"]["storage"] is True
    assert body["checks"]["ffmpeg"] is True


def test_correlation_id_echoed(client):
    resp = client.get("/health", headers={"X-Correlation-Id": "custom-trace-999"})
    assert resp.headers["X-Correlation-Id"] == "custom-trace-999"


def test_openapi_yaml_matches_live_service(client):
    spec_path = Path(__file__).resolve().parents[1] / "openapi.yaml"
    assert spec_path.exists(), "openapi.yaml must exist at repository root"
    static_spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    live_spec = client.get("/openapi.json").json()

    assert static_spec["openapi"].startswith("3.1")
    assert set(static_spec["paths"].keys()) == set(live_spec["paths"].keys())
    for p in live_spec["paths"]:
        if p not in ("/health", "/ready"):
            assert p.startswith("/api/v1/"), f"Non-observability route must be under /api/v1/: {p}"


def test_idempotency_key_replays_same_job_and_detects_conflict(client):
    payload = {
        "script": "Consistent daily habits create lasting progress for creative teams.",
        "aspect_ratios": ["16:9"],
        "captions": {"burn_in": False, "srt": True},
    }
    headers = {"Idempotency-Key": "idem-test-001"}
    r1 = client.post("/api/v1/broll/jobs", json=payload, headers=headers)
    assert r1.status_code == 202
    j1 = r1.json()
    assert j1["idempotent_replay"] is False

    # Same key + same payload -> exact same job_id!
    r2 = client.post("/api/v1/broll/jobs", json=payload, headers=headers)
    assert r2.status_code == 202
    j2 = r2.json()
    assert j2["job_id"] == j1["job_id"]
    assert j2["idempotent_replay"] is True

    # Same key + different payload -> 409 IDEMPOTENCY_CONFLICT
    diff_payload = {**payload, "script": "A completely different narration script."}
    r3 = client.post("/api/v1/broll/jobs", json=diff_payload, headers=headers)
    assert r3.status_code == 409
    err = r3.json()["error"]
    assert err["code"] == "IDEMPOTENCY_CONFLICT"
    assert "message" in err and "retryable" in err and "details" in err

    # Wait for background job to finish cleanly so temp dirs stay clean
    poll_until_done(client, j1["job_id"])


def test_standardized_error_envelope_on_empty_script_and_missing_job(client):
    r_empty = client.post("/api/v1/broll/jobs", json={"script": "   ", "aspect_ratios": ["16:9"]})
    assert r_empty.status_code == 422
    err = r_empty.json()["error"]
    assert err["code"] == "EMPTY_SCRIPT"
    assert isinstance(err["retryable"], bool)

    r_missing = client.get("/api/v1/jobs/job_does_not_exist")
    assert r_missing.status_code == 404
    err2 = r_missing.json()["error"]
    assert err2["code"] == "JOB_NOT_FOUND"


def test_cancel_job_endpoint(client):
    payload = {
        "script": "Fast cancellation prevents wasted queue wait times for creators.",
        "aspect_ratios": ["16:9"],
        "captions": {"burn_in": False, "srt": True},
    }
    r = client.post("/api/v1/broll/jobs", json=payload)
    assert r.status_code == 202
    jid = r.json()["job_id"]

    rc = client.post(f"/api/v1/jobs/{jid}/cancel")
    assert rc.status_code == 200
    assert rc.json()["status"] in ("cancelled", "succeeded", "failed")

    status_resp = client.get(f"/api/v1/jobs/{jid}")
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] in ("failed", "succeeded")
