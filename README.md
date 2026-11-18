# IncuBrix Script & Voiceover to B-roll Video Engine

A containerized HTTP API service that converts voiceovers (`.mp3`, `.wav`, `.m4a`, up to 10 minutes), plain-text scripts, or IncuBrix scene lists (`[{"text": "...", "visual_hint": "..."}]`) into finished, licence-clean, caption-synced B-roll videos across `16:9`, `9:16`, and `1:1` aspect ratios.

---

## Architecture

```
Client / Batch CLI / Web UI
        │  POST /api/v1/broll/jobs (+ Idempotency-Key, X-Correlation-Id)
        ▼
┌─────────────────────────────────────────┐
│ FastAPI Service (OpenAPI 3.1)           │ ──► PostgreSQL / SQLite (Jobs + Idempotency)
│ /health  /ready  /api/v1/*  / (Demo UI) │ ──► Redis / Inline Worker Queue
└─────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ Background Worker Pipeline (Swappable Commercial-Use Adapters)           │
│  1. Ingest & Validate  : Magic-byte + ffprobe + silence/duration checks  │
│  2. Narration & Timing : Gemini 3.8 Flash-Lite TTS + Groq Whisper ASR    │
│  3. Semantic Planner   : Gemini 3.5 Flash-Lite (sentence boundaries,     │
│                          homonym disambiguation, safety/brand/sad rules) │
│  4. Footage Router     : Pexels + Pixabay + Gemini Image Ken-Burns +     │
│                          Programmatic Animated Stat Cards (circuit-break)│
│  5. Independent VLM    : Groq Llama-4 Vision 1-5 scoring + avoid check + │
│                          subject_focus_x for 9:16 edge-subject framing   │
│  6. Normalizer & Cache : FFmpeg 1080p/30fps/yuv420p, color/luma match,   │
│                          per-scene cache (fast swap-a-clip), Noto SRT    │
└──────────────────────────────────────────────────────────────────────────┘
        │
        ▼
MinIO / S3 Object Storage  (video_{ratio}.mp4, captions.srt, timeline.json, manifest.json)
        │
        ▼ (optional)
HMAC-SHA256 Signed Callback Webhook (X-IncuBrix-Signature)
```

---

## Quick Start (Under 10 Commands)

### Option A: Full Stack with Docker Compose (Integration Standard #1)
```bash
cp .env.example .env
docker compose up --build -d
curl http://localhost:8000/ready
# Run all unit, integration, and E2E tests offline with ZERO API keys:
docker compose exec api pytest -v
```
Open the Live Demo UI at `http://localhost:8000/` and interactive OpenAPI 3.1 docs at `http://localhost:8000/docs`.

### Option B: Local Python Environment (Zero API Keys in Mock Mode)
```bash
conda create -n incubrix -c conda-forge python=3.12 ffmpeg git -y
conda activate incubrix
pip install -e ".[dev]"
pytest -v
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Switching to Live AI & Stock Media Mode
Set `ADAPTER_MODE=live` and provide your free-tier commercial-use keys in `.env`:
```env
ADAPTER_MODE=live
GEMINI_API_KEY=...
GROQ_API_KEY=...
PEXELS_API_KEY=...
PIXABAY_API_KEY=...
```

---

## Batch Runner CLI (Integration Standard #10)
Run an entire folder or `.jsonl` test pack against any local or deployed instance and download all outputs + `batch_summary.json` into one folder:
```bash
python -m app.cli.batch \
  --input test_pack/inputs \
  --api-url http://localhost:8000 \
  --output test_pack/outputs
```
Pass `--token <API_ACCESS_TOKEN>` when targeting a token-protected live demo.

---

## Multi-Source Selection Rule (Advanced Level)
For every scene, the orchestrator records the exact rule (`selection_rule`) and measured rationale (`selection_reason`) in `timeline.json`:
1. **`R1_PREFERRED_SOURCE`**: Used when a user explicitly requests a source during single-scene clip swap (`POST /api/v1/broll/jobs/{job_id}/scenes/{scene_index}/swap`).
2. **`R2_STOCK_THRESHOLD_MET`**: Stock sources (`pexels`, `pixabay`) are searched first with lexical + VLM `avoid` filtering and ID + perceptual `dHash` deduplication. If the best stock clip achieves independent VLM score `>= STOCK_MIN_RELEVANCE_SCORE` (default `3.5 / 5.0`), it is selected (`$0` cost, real-world footage).
3. **`R3_GENERATIVE_ESCALATION`**: If the best stock score is `< 3.5 / 5.0` (or the scene is an abstract metaphor) **and** no real named person or brand is present (`block_ai_generation == false`), the engine generates a bespoke visual via Gemini Image + 30fps Ken Burns motion (`ai_generated: true`).
4. **`R4_STAT_GRAPHIC` / `R4_RESILIENT_FALLBACK`**: For quantitative statistics with weak stock fit (`< 4.10`), or if all external sources fail/are unreachable, the engine renders an original animated motion-graphic card (`statcard`) so the job never crashes silently.

---

## Error Codes (Integration Standard #5)
All errors return `{"error": {"code", "message", "retryable", "details"}}`:

| Code | HTTP | Retryable | Meaning |
|---|---|---|---|
| `INVALID_INPUT` | 400 / 422 | No | Payload failed schema validation or invalid URL/JSON metadata. |
| `EMPTY_SCRIPT` | 400 / 422 | No | Script or scenes list contains no speakable text. |
| `UNSUPPORTED_MEDIA_TYPE` | 415 | No | File is not a valid `.mp3`, `.wav`, or `.m4a` audio stream. |
| `AUDIO_TOO_LONG` | 400 | No | Audio exceeds `MAX_AUDIO_DURATION_SECONDS` (default 600s). |
| `AUDIO_SILENT_OR_NO_SPEECH` | 422 | No | Audio is silent or contains no intelligible speech (e.g. music-only). |
| `AUDIO_DOWNLOAD_FAILED` | 400 | Maybe | Could not fetch audio from `audio_url`. |
| `AUDIO_CORRUPTED` | 400 | No | `ffprobe` failed to decode the audio file. |
| `JOB_NOT_FOUND` | 404 | No | Unknown `job_id`. |
| `SCENE_NOT_FOUND` | 404 | No | `scene_index` out of bounds for the job timeline. |
| `JOB_NOT_READY` | 409 | No | Cannot swap a clip before job reaches `succeeded`. |
| `IDEMPOTENCY_CONFLICT` | 409 | No | Same `Idempotency-Key` reused with a different request payload. |
| `UNAUTHORIZED` | 401 | No | Missing or invalid Bearer token when `API_ACCESS_TOKEN` is set. |
| `RATE_LIMIT_EXCEEDED` | 429 | Yes | Exceeded `RATE_LIMIT_PER_MINUTE` POST requests per IP. |
| `PROVIDER_UNAVAILABLE` | 503 | Yes | External provider unreachable or returned HTTP error. |
| `RENDER_FAILED` | 500 | Yes | FFmpeg normalization or concatenation error. |
| `STORAGE_ERROR` | 404 / 502 | Maybe | Object storage read/write error. |
| `INTERNAL_ERROR` | 500 | Yes | Unexpected server error. |

---

## Configuration Reference
See [`.env.example`](.env.example) for the complete list of environment variables (zero hardcoded paths, model names, or secrets).
