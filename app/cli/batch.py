"""CLI Batch Runner (Integration Standard #10): runs a folder or JSONL of inputs against the API."""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List
import uuid

import httpx


def load_batch_inputs(input_path: Path) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    if input_path.is_file():
        if input_path.suffix.lower() == ".jsonl":
            for idx, line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), start=1):
                if line.strip():
                    item = json.loads(line)
                    item.setdefault("name", f"case_{idx:02d}")
                    cases.append(item)
        else:
            data = json.loads(input_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for idx, item in enumerate(data, start=1):
                    item.setdefault("name", f"case_{idx:02d}")
                    cases.append(item)
            else:
                data.setdefault("name", input_path.stem)
                cases.append(data)
    elif input_path.is_dir():
        for fp in sorted(input_path.glob("*.json")):
            item = json.loads(fp.read_text(encoding="utf-8"))
            item.setdefault("name", fp.stem)
            cases.append(item)
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    return cases


def run_batch(
    input_path: Path,
    api_url: str,
    output_dir: Path,
    token: str = "",
    poll_interval: float = 1.5,
    timeout_per_job: float = 300.0,
) -> Dict[str, Any]:
    cases = load_batch_inputs(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_url = api_url.rstrip("/")
    headers: Dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    summary_rows: List[Dict[str, Any]] = []
    with httpx.Client(timeout=60.0) as client:
        for case in cases:
            case_name = str(case.get("name") or f"case_{uuid.uuid4().hex[:6]}")
            case_dir = output_dir / case_name
            case_dir.mkdir(parents=True, exist_ok=True)
            corr_id = f"batch_{case_name}_{uuid.uuid4().hex[:6]}"
            idem_key = f"batch_{case_name}_{uuid.uuid4().hex[:8]}"
            req_headers = {**headers, "X-Correlation-Id": corr_id, "Idempotency-Key": idem_key}

            t0 = time.perf_counter()
            audio_file_path = case.get("audio_file")
            payload = {k: v for k, v in case.items() if k not in ("name", "audio_file")}

            try:
                if audio_file_path:
                    fp = Path(audio_file_path)
                    if not fp.is_absolute():
                        fp = (input_path.parent if input_path.is_file() else input_path) / fp
                    with fp.open("rb") as f:
                        resp = client.post(
                            f"{base_url}/api/v1/broll/jobs/upload",
                            headers=req_headers,
                            files={"audio_file": (fp.name, f, "application/octet-stream")},
                            data={"metadata": json.dumps(payload)},
                        )
                else:
                    resp = client.post(f"{base_url}/api/v1/broll/jobs", headers=req_headers, json=payload)

                if resp.status_code != 202:
                    err_body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {"raw": resp.text}
                    (case_dir / "error.json").write_text(json.dumps(err_body, indent=2), encoding="utf-8")
                    summary_rows.append(
                        {"name": case_name, "status": "rejected", "http_status": resp.status_code, "error": err_body}
                    )
                    print(f"[REJECTED] {case_name} -> HTTP {resp.status_code}")
                    continue

                job_id = resp.json()["job_id"]
                final_status = None
                while time.perf_counter() - t0 < timeout_per_job:
                    st_resp = client.get(f"{base_url}/api/v1/jobs/{job_id}", headers=headers)
                    st = st_resp.json()
                    if st.get("status") in ("succeeded", "failed"):
                        final_status = st
                        break
                    time.sleep(poll_interval)

                if final_status is None:
                    summary_rows.append({"name": case_name, "job_id": job_id, "status": "timeout"})
                    print(f"[TIMEOUT] {case_name} ({job_id})")
                    continue

                (case_dir / "job_status.json").write_text(json.dumps(final_status, indent=2), encoding="utf-8")

                # Download all produced artifacts
                res = final_status.get("result") or {}
                for ratio, dl_url in (res.get("videos") or {}).items():
                    tag = ratio.replace(":", "x")
                    r = client.get(dl_url, headers=headers)
                    if r.status_code == 200:
                        (case_dir / f"video_{tag}.mp4").write_bytes(r.content)
                for field_name, fname in (
                    ("captions_srt", "captions.srt"),
                    ("timeline_json", "timeline.json"),
                    ("manifest_json", "manifest.json"),
                ):
                    dl_url = res.get(field_name)
                    if dl_url:
                        r = client.get(dl_url, headers=headers)
                        if r.status_code == 200:
                            (case_dir / fname).write_bytes(r.content)

                elapsed = round(time.perf_counter() - t0, 2)
                summary_rows.append(
                    {
                        "name": case_name,
                        "job_id": job_id,
                        "status": final_status.get("status"),
                        "elapsed_seconds": elapsed,
                        "warnings": final_status.get("warnings", []),
                        "metrics": final_status.get("metrics"),
                        "error": final_status.get("error"),
                    }
                )
                print(f"[{final_status.get('status', 'unknown').upper()}] {case_name} ({job_id}) in {elapsed}s")
            except Exception as exc:
                summary_rows.append({"name": case_name, "status": "client_error", "error": str(exc)})
                print(f"[ERROR] {case_name}: {exc}")

    summary = {
        "api_url": base_url,
        "total_cases": len(cases),
        "succeeded": sum(1 for r in summary_rows if r.get("status") == "succeeded"),
        "failed_or_rejected": sum(1 for r in summary_rows if r.get("status") != "succeeded"),
        "cases": summary_rows,
    }
    (output_dir / "batch_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="IncuBrix B-roll Engine CLI Batch Runner")
    parser.add_argument("--input", required=True, type=Path, help="Folder of .json files, single .json, or .jsonl file")
    parser.add_argument("--api-url", default="http://localhost:8000", help="Base URL of running API")
    parser.add_argument("--output", required=True, type=Path, help="Output directory for all artifacts and summary")
    parser.add_argument("--token", default="", help="Optional Bearer API access token")
    parser.add_argument("--timeout", type=float, default=300.0, help="Per-job timeout seconds")
    args = parser.parse_args()

    summary = run_batch(
        input_path=args.input,
        api_url=args.api_url,
        output_dir=args.output,
        token=args.token,
        timeout_per_job=args.timeout,
    )
    print(f"\nBatch complete: {summary['succeeded']}/{summary['total_cases']} succeeded -> {args.output / 'batch_summary.json'}")
    sys.exit(0 if summary["total_cases"] > 0 else 1)


if __name__ == "__main__":
    main()
