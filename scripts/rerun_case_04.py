import json
from pathlib import Path
import time
import httpx

base_url = "http://127.0.0.1:8000"
case_name = "04_story_craft_trust_en_gb"
case_dir = Path("test_pack/outputs") / case_name
case_dir.mkdir(parents=True, exist_ok=True)

payload = {
    "script": "Trust is earned slowly through quiet craftsmanship and kept promises. In a small workshop beside the river bank, a woodworker shapes every joint by hand without rushing. Years later, those sturdy tables still gather families together every evening.",
    "voice": "en-GB-female",
    "aspect_ratios": ["16:9", "9:16"],
    "style": "warm cinematic documentary natural light",
    "avoid": [],
    "captions": {"burn_in": True, "srt": True},
}

headers = {
    "X-Correlation-Id": f"batch_{case_name}_rerun",
    "Idempotency-Key": f"batch_{case_name}_rerun_key",
}

print(f"Submitting {case_name}...")
with httpx.Client(timeout=180.0) as client:
    resp = client.post(f"{base_url}/api/v1/broll/jobs", headers=headers, json=payload)
    print("Submit HTTP status:", resp.status_code)
    job_id = resp.json()["job_id"]
    print("Job ID:", job_id)

    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 300:
        st = client.get(f"{base_url}/api/v1/jobs/{job_id}").json()
        status = st.get("status")
        stage = st.get("stage")
        progress = st.get("progress")
        print(f"[{time.perf_counter() - t0:.1f}s] status={status}, stage={stage}, progress={progress}%")
        if status in ("succeeded", "failed"):
            final_status = st
            break
        time.sleep(3)

    if final_status.get("status") == "succeeded":
        print("Success! Downloading artifacts...")
        (case_dir / "job_status.json").write_text(json.dumps(final_status, indent=2), encoding="utf-8")
        res = final_status.get("result") or {}
        for ratio, dl_url in (res.get("videos") or {}).items():
            tag = ratio.replace(":", "x")
            r = client.get(dl_url)
            if r.status_code == 200:
                (case_dir / f"video_{tag}.mp4").write_bytes(r.content)
                print(f"  Saved video_{tag}.mp4 ({len(r.content)} bytes)")
        for field, fname in [("captions_srt", "captions.srt"), ("timeline_json", "timeline.json"), ("manifest_json", "manifest.json")]:
            url = res.get(field)
            if url:
                r = client.get(url)
                if r.status_code == 200:
                    (case_dir / fname).write_bytes(r.content)
                    print(f"  Saved {fname}")
        print("Case 04 rerun complete and saved!")
    else:
        print("Job failed:", final_status.get("error"))
