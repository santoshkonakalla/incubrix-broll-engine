# AI Coding Tools Disclosure (`AI_USE.md`)

In accordance with IncuBrix Assessment 3 transparency guidelines, all AI coding assistants and AI models used during development and testing are disclosed below.

## 1. AI Coding Assistants Used

| Tool | Model(s) Used | How It Was Used |
|---|---|---|
| Google Antigravity IDE Agent | Gemini / Claude | Architecture design review, boilerplate scaffolding for FastAPI/Docker/tests, FFmpeg filter graph crafting, debugging, test script authoring, and documentation drafting. |

## 2. Human Ownership & Verification
- Every architectural decision (swappable adapter boundaries, deterministic ASR sentence-boundary pacing, explicit 4-rule multi-source footage router, per-scene cached `-c:v copy` swap-a-clip design, safety/brand/homonym guardrails, and independent VLM evaluation) was designed, reviewed, tested, and verified by the candidate.
- All automated tests, live API runs, manual 1–5 human relevance ratings, turnaround/cost/concurrency benchmarks, and hard-case checks were executed directly against the real service and recorded without fabrication.
- The candidate understands every module and can walk through, debug, and modify any part of the codebase during the live evaluation session.
