"""Production benchmark: cold/warm 60s turnaround, swap speedup, and concurrent job capacity (Outstanding Level)."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from typing import Any, Dict, List
import uuid

import httpx

SCRIPT_60S = (
    "Saving money consistently is less about drastic sacrifice and more about clear everyday systems. "
    "Start by tracking where every rupee goes during your first thirty days using a clean budget sheet. "
    "When you automate a monthly transfer into a separate savings account on payday, discipline becomes effortless. "
    "Modern cloud banking apps let households review recurring subscriptions and cancel silent leaks in seconds. "
    "Even modest returns compound meaningfully over a decade when combined with steady patience. "
    "Remember that real financial value comes from peace of mind and freedom of choice, not just numbers on a screen. "
    "In 2025, surveys showed automated savers kept 40% more of their annual income than manual planners. "
    "Start small this week, stay consistent every month, and let compounding do the heavy lifting for your future."
)


def _submit_and_wait(base_url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.perf_counter()
    with httpx.Client(timeout=360.0) as client:
        r = client.post(
            f"{base_url}/api/v1/broll/jobs",
            headers={**headers, "Idempotency-Key": f"bench_{uuid.uuid4().hex[:10]}"},
            json=payload,
        )
        r.raise_for_status()
        job_id = r.json()["job_id"]
        while True:
            st = client.get(f"{base_url}/api/v1/jobs/{job_id}", headers=headers).json()
            if st["status"] in ("succeeded", "failed"):
                st["_wall_seconds"] = round(time.perf_counter() - t0, 3)
                return st
            time.sleep(0.5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark turnaround, swap speedup, and concurrency")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--token", default="")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("test_pack/benchmarks.json"))
    args = parser.parse_args()

    base_url = args.api_url.rstrip("/")
    headers = {"Authorization": f"Bearer {args.token}"} if args.token else {}
    payload = {
        "script": SCRIPT_60S,
        "voice": "en-IN-female",
        "aspect_ratios": ["16:9", "9:16"],
        "captions": {"burn_in": True, "srt": True},
    }

    print("1/3 Running cold 60s turnaround test...")
    cold = _submit_and_wait(base_url, headers, payload)
    print(f"    Cold: {cold['_wall_seconds']}s (status={cold['status']})")

    print("2/3 Running warm 60s turnaround + Swap-a-Clip speedup test...")
    warm = _submit_and_wait(base_url, headers, payload)
    print(f"    Warm: {warm['_wall_seconds']}s (status={warm['status']})")

    swap_res = {}
    if warm["status"] == "succeeded":
        with httpx.Client(timeout=120.0) as client:
            sr = client.post(
                f"{base_url}/api/v1/broll/jobs/{warm['job_id']}/scenes/0/swap",
                headers=headers,
                json={"query": "sunlit desk notebook budgeting pen"},
            )
            swap_res = sr.json()
        print(f"    Swap-a-clip: {swap_res.get('swap_duration_seconds')}s vs full render {swap_res.get('original_full_render_seconds')}s")

    print(f"3/3 Running concurrency test ({args.concurrency} simultaneous jobs)...")
    t_conc = time.perf_counter()
    short_payload = {
        "script": "Consistent creative workflows help teams ship clear educational videos faster every single week.",
        "voice": "en-IN-female",
        "aspect_ratios": ["16:9"],
        "captions": {"burn_in": True, "srt": True},
    }
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(_submit_and_wait, base_url, headers, short_payload) for _ in range(args.concurrency)]
        conc_results: List[Dict[str, Any]] = [f.result() for f in futures]
    conc_wall = round(time.perf_counter() - t_conc, 3)

    report = {
        "api_url": base_url,
        "cold_60s": {"job_id": cold["job_id"], "status": cold["status"], "wall_seconds": cold["_wall_seconds"], "metrics": cold.get("metrics")},
        "warm_60s": {"job_id": warm["job_id"], "status": warm["status"], "wall_seconds": warm["_wall_seconds"], "metrics": warm.get("metrics")},
        "swap_a_clip": swap_res,
        "concurrency": {
            "parallel_jobs": args.concurrency,
            "all_succeeded": all(r["status"] == "succeeded" for r in conc_results),
            "total_wall_seconds": conc_wall,
            "per_job_wall_seconds": [r["_wall_seconds"] for r in conc_results],
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved benchmark report -> {args.out}")


if __name__ == "__main__":
    main()
