"""End-to-end tests: script, audio upload, IncuBrix scenes, fast swap-a-clip, failover, and all 7 hard cases."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import struct
import wave

from app.adapters.mock_adapters import synthesize_speech_like_wav
from tests.conftest import poll_until_done


def _make_silent_wav(path: Path, seconds: float = 3.0, sr: int = 16000) -> Path:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(b"\x00\x00" * int(seconds * sr))
    return path


def test_e2e_script_multi_ratio_captions_qa_and_fast_swap(client):
    payload = {
        "script": (
            "Trust is earned slowly through steady everyday actions. "
            "Modern cloud computing helped our sales grew 40% in 2025 across global teams. "
            "Learning Python lets engineers automate boring tasks in minutes."
        ),
        "voice": "en-IN-female",
        "aspect_ratios": ["16:9", "9:16"],
        "style": "bright, modern, real-world footage",
        "avoid": ["alcohol"],
        "captions": {"burn_in": True, "srt": True},
    }
    resp = client.post("/api/v1/broll/jobs", json=payload, headers={"Idempotency-Key": "e2e-script-1"})
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    status = poll_until_done(client, job_id)
    assert status["status"] == "succeeded", f"Job failed: {status}"
    assert status["progress"] == 100
    res = status["result"]
    assert "16:9" in res["videos"] and "9:16" in res["videos"]

    # Download and validate timeline.json + manifest.json + captions.srt + MP4s
    tl = client.get(res["timeline_json"]).json()
    mf = client.get(res["manifest_json"]).json()
    srt_text = client.get(res["captions_srt"]).text

    assert tl["repeat_clip_count"] == 0
    assert tl["scene_count"] >= 2
    assert tl["scenes"][0]["start_time"] == 0.0
    assert abs(tl["scenes"][-1]["end_time"] - tl["total_duration"]) <= 0.15
    assert "00:00:" in srt_text

    # Every clip must be licence-clean and have attribution + URL + rule reason
    for sc in tl["scenes"]:
        assert sc["licence"]
        assert sc["attribution"]
        assert sc["source_url"]
        assert sc["selection_rule"]
        assert sc["selection_reason"]
        assert 1.0 <= sc["relevance_score"] <= 5.0

    # Automated FFmpeg QA checks must pass for both aspect ratios
    assert mf["qa_checks"]["16:9"]["passed"] is True
    assert mf["qa_checks"]["9:16"]["passed"] is True
    assert mf["qa_checks"]["16:9"]["black_frame_events"] == 0
    assert mf["qa_checks"]["16:9"]["freeze_frame_events"] == 0

    # Verify video downloads are non-empty valid MP4s
    v16 = client.get(res["videos"]["16:9"])
    assert v16.status_code == 200 and len(v16.content) > 10_000

    # --- Advanced Level: Fast Swap-a-Clip ---
    old_clip_id = tl["scenes"][0]["clip_id"]
    swap_resp = client.post(
        f"/api/v1/broll/jobs/{job_id}/scenes/0/swap",
        json={"query": "sunlit architecture glass bridge wide shot", "preferred_source": "pixabay"},
    )
    assert swap_resp.status_code == 200, swap_resp.text
    swap_data = swap_resp.json()
    assert swap_data["new_clip_id"] != old_clip_id
    assert swap_data["re_rendered_scenes"] == 1
    assert swap_data["cached_scenes_reused"] == tl["scene_count"] - 1

    # Confirm timeline updated with zero repeats
    tl_after = client.get(res["timeline_json"]).json()
    assert tl_after["scenes"][0]["clip_id"] == swap_data["new_clip_id"]
    assert len({s["clip_id"] for s in tl_after["scenes"]}) == len(tl_after["scenes"])


def test_e2e_audio_upload_and_incubrix_scenes_and_multilingual(client, tmp_path: Path):
    # 1. Audio upload (WAV)
    wav_path, _ = synthesize_speech_like_wav(
        "Morning coffee fuels focus while developers review clean analytics dashboards.",
        tmp_path / "voiceover.wav",
    )
    with wav_path.open("rb") as f:
        r_up = client.post(
            "/api/v1/broll/jobs/upload",
            files={"audio_file": ("voiceover.wav", f, "audio/wav")},
            data={"metadata": '{"aspect_ratios": ["16:9"], "captions": {"burn_in": false, "srt": true}}'},
            headers={"Idempotency-Key": "e2e-upload-1"},
        )
    assert r_up.status_code == 202
    st_up = poll_until_done(client, r_up.json()["job_id"])
    assert st_up["status"] == "succeeded"

    # 2. IncuBrix Scene List + Hindi narration
    r_sc = client.post(
        "/api/v1/broll/jobs",
        json={
            "scenes": [
                {"text": "आज हम स्मार्ट बचत के आसान तरीके सीखेंगे।", "visual_hint": "family planning household budget warm home"},
                {"text": "नियमित निवेश से भविष्य सुरक्षित बनता है।", "visual_hint": "calm sunrise over green fields hopeful future"},
            ],
            "voice": "hi-IN-female",
            "aspect_ratios": ["16:9"],
            "captions": {"burn_in": True, "srt": True},
        },
        headers={"Idempotency-Key": "e2e-scenes-hi"},
    )
    assert r_sc.status_code == 202
    st_sc = poll_until_done(client, r_sc.json()["job_id"])
    assert st_sc["status"] == "succeeded"
    tl_hi = client.get(st_sc["result"]["timeline_json"]).json()
    assert tl_hi["language"] == "hi"
    assert tl_hi["scene_count"] == 2
    assert tl_hi["scenes"][0]["visual_hint"] == "family planning household budget warm home"


def test_e2e_resilient_failover_when_sources_disabled(client):
    # Simulate Pexels + Pixabay + Generative all offline -> must gracefully complete via statcard fallback!
    resp = client.post(
        "/api/v1/broll/jobs",
        json={
            "script": "Even when external stock providers experience an outage, resilient pipelines complete cleanly.",
            "aspect_ratios": ["16:9"],
            "disabled_sources": ["pexels", "pixabay", "generative"],
            "captions": {"burn_in": False, "srt": True},
        },
        headers={"Idempotency-Key": "e2e-failover-1"},
    )
    assert resp.status_code == 202
    st = poll_until_done(client, resp.json()["job_id"])
    assert st["status"] == "succeeded"
    tl = client.get(st["result"]["timeline_json"]).json()
    assert all(s["source"] == "statcard" for s in tl["scenes"])


def test_hard_cases_invalid_inputs_and_concurrent_users(client, tmp_path: Path):
    # Wrong file type (.txt) -> 415 UNSUPPORTED_MEDIA_TYPE
    r_bad = client.post(
        "/api/v1/broll/jobs/upload",
        files={"audio_file": ("notes.txt", b"hello not audio", "text/plain")},
        data={"metadata": '{"aspect_ratios": ["16:9"]}'},
    )
    assert r_bad.status_code == 415
    assert r_bad.json()["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"

    # Silent audio -> job fails clearly with AUDIO_SILENT_OR_NO_SPEECH
    silent_wav = _make_silent_wav(tmp_path / "silent.wav")
    with silent_wav.open("rb") as f:
        r_sil = client.post(
            "/api/v1/broll/jobs/upload",
            files={"audio_file": ("silent.wav", f, "audio/wav")},
            data={"metadata": '{"aspect_ratios": ["16:9"]}'},
        )
    assert r_sil.status_code == 202
    st_sil = poll_until_done(client, r_sil.json()["job_id"])
    assert st_sil["status"] == "failed"
    assert st_sil["error"]["code"] == "AUDIO_SILENT_OR_NO_SPEECH"

    # Named person + brand + sad news -> completes safely with warnings logged!
    r_safe = client.post(
        "/api/v1/broll/jobs",
        json={
            "script": (
                "Elon Musk and Apple executives commented after the tragic flood disaster. "
                "Communities across the region are rebuilding with quiet resilience."
            ),
            "aspect_ratios": ["9:16"],
            "captions": {"burn_in": False, "srt": True},
        },
    )
    assert r_safe.status_code == 202
    st_safe = poll_until_done(client, r_safe.json()["job_id"])
    assert st_safe["status"] == "succeeded"
    assert len(st_safe["warnings"]) >= 2
    tl_safe = client.get(st_safe["result"]["timeline_json"]).json()
    # Ensure NO AI-generated face/logo clip was produced for the person/brand scene
    assert tl_safe["scenes"][0]["ai_generated"] is False

    # Two users submitting concurrently -> both succeed without corrupting each other
    def submit_one(tag: str):
        r = client.post(
            "/api/v1/broll/jobs",
            json={
                "script": f"Concurrent user {tag} creating a bright modern explainer video for creators.",
                "aspect_ratios": ["16:9"],
                "captions": {"burn_in": False, "srt": True},
            },
        )
        assert r.status_code == 202
        return poll_until_done(client, r.json()["job_id"])

    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(submit_one, "alpha")
        f2 = pool.submit(submit_one, "beta")
        res1, res2 = f1.result(), f2.result()

    assert res1["status"] == "succeeded"
    assert res2["status"] == "succeeded"
    assert res1["job_id"] != res2["job_id"]
