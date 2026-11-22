import sys
import time
import httpx

payload = {
    "script": (
        "Learning Python transforms how you solve everyday engineering problems. "
        "Modern cloud computing servers run pipelines automatically."
    ),
    "voice": "en-IN-female",
    "aspect_ratios": ["16:9"],
    "style": "bright modern footage",
    "avoid": ["alcohol"],
    "captions": {"burn_in": True, "srt": True},
}

resp = httpx.post("http://127.0.0.1:8000/api/v1/broll/jobs", json=payload, timeout=30.0)
print("Accepted:", resp.status_code, resp.json())
if resp.status_code != 202:
    sys.exit(1)

job_id = resp.json()["job_id"]

for i in range(120):
    time.sleep(2)
    st = httpx.get(f"http://127.0.0.1:8000/api/v1/jobs/{job_id}", timeout=10.0).json()
    stage = st.get("stage")
    progress = st.get("progress")
    status = st.get("status")
    print(f"[{i*2}s] stage={stage} progress={progress}% status={status}")
    if status == "succeeded":
        print("\nSUCCESS!")
        print("Result:", st.get("result"))
        print("Metrics:", st.get("metrics"))
        break
    elif status == "failed":
        print("\nFAILED!")
        print("Error:", st.get("error"))
        sys.exit(2)
