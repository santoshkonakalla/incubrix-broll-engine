# Sources, Models, APIs, Libraries, and Licences (`SOURCES.md`)

Everything in the default pipeline path and everything it generates is licensed for commercial use. No CC BY-NC, research-only, watermarked, or AGPL dependencies are used in the shipped service.

## 1. AI Models & Hosted APIs

| Role | Provider | Model / API | Commercial Use Licence | Notes |
|---|---|---|---|---|
| Semantic Scene Planner | Google Gemini API | `gemini-3.5-flash-lite` | Google Gemini API Terms of Service (commercial use permitted) | Structured JSON planning, homonym disambiguation, multilingual visual query translation |
| Text-to-Speech (EN/HI/TA) | Google Gemini API | `gemini-3.8-flash-lite-tts` | Google Gemini API Terms of Service (commercial use permitted) | Native WAV synthesis for English (`en-IN`, `en-US`), Hindi (`hi-IN`), and Tamil (`ta-IN`) |
| Generative B-roll Visuals | Google Gemini API | `gemini-3.1-flash-lite-image` | Google Gemini API Terms of Service (commercial use permitted) | Image generation animated via FFmpeg Ken Burns motion; always flagged `ai_generated: true` |
| Speech-to-Text (ASR) | Groq Cloud API | `whisper-large-v3-turbo` | OpenAI Whisper weights (MIT) + Groq Commercial API Terms | Word-level timestamps for English, Hindi, and Tamil; no-speech detection |
| Independent VLM Scorer | Groq Cloud API | `meta-llama/llama-4-scout-17b-16e-instruct` | Llama 4 Community License (commercial use permitted under threshold) + Groq Terms | Independent model family from Gemini planner; 1–5 relevance rating, avoid-list check, subject focus X |

## 2. Stock Media & Generated Media Sources

| Source | Licence | Commercial Use | Watermarks | Attribution Recorded |
|---|---|---|---|---|
| Pexels Video API | Pexels License (`https://www.pexels.com/license/`) | Yes | None | Yes (`source_url`, contributor name, licence in `timeline.json` & `manifest.json`) |
| Pixabay Video API | Pixabay Content License (`https://pixabay.com/service/license-summary/`) | Yes | None | Yes (`source_url`, contributor name, licence in `timeline.json` & `manifest.json`) |
| Programmatic Stat & Fallback Cards | Original code (MIT) | Yes | None | Generated in-process with Pillow + OpenCV + FFmpeg; flagged `ai_generated: true` |

## 3. Core Open-Source Libraries & System Tools

| Component | Licence | Purpose |
|---|---|---|
| FastAPI & Starlette | MIT / BSD-3-Clause | OpenAPI 3.1 HTTP API framework |
| Uvicorn | BSD-3-Clause | ASGI web server |
| Pydantic & pydantic-settings | MIT | Request/response validation & env-driven configuration |
| SQLAlchemy | MIT | PostgreSQL / SQLite job & idempotency store |
| Redis-py & Redis 7 | MIT / RSALv2-SSPL (Docker service) | Background job queue |
| Boto3 & MinIO | Apache-2.0 | S3-compatible object storage |
| Structlog | MIT / Apache-2.0 | Structured JSON logging |
| HTTPX & Tenacity | BSD-3-Clause / Apache-2.0 | HTTP client for APIs & webhooks |
| OpenCV (`opencv-python-headless`) | Apache-2.0 | Saliency-based smart crop, perceptual dHash, color stats, keyframe extraction |
| Pillow | HPND (MIT-like) | Typography & animated data-card rendering |
| NumPy & SciPy | BSD-3-Clause | Numerical processing & VLM-human correlation analysis |
| `google-genai` | Apache-2.0 | Official Google Gemini Interactions SDK |
| FFmpeg / FFprobe | LGPL/GPL CLI invocation | Invoked as external CLI subprocess for video normalization, concat, captions & QA |
| Google Noto Fonts (`fonts-noto-core`, `fonts-noto-extra`) | SIL Open Font License 1.1 | Commercial-free Latin, Devanagari (Hindi), and Tamil subtitle rendering |
| Pytest | MIT | Automated unit, integration, and end-to-end test suite |
