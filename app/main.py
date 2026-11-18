"""FastAPI application entrypoint, middleware, health/ready probes, and OpenAPI 3.1 config."""

from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
import shutil
import time
from typing import Deque, Dict
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from app import __version__
from app.api.routes import router as v1_router
from app.config import get_settings
from app.db import check_db_ready, init_db
from app.errors import (
    AppError,
    app_error_handler,
    http_exception_handler,
    unhandled_exception_handler,
    validation_error_handler,
)
from app.logging_config import configure_logging, correlation_id_ctx, get_logger
from app.queue import get_queue
from app.schemas import HealthResponse, ReadyResponse
from app.storage import get_storage

logger = get_logger("main")


class SlidingWindowRateLimiter:
    def __init__(self) -> None:
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)

    def allow(self, client_key: str, max_per_minute: int) -> bool:
        now = time.monotonic()
        bucket = self._hits[client_key]
        while bucket and now - bucket[0] > 60.0:
            bucket.popleft()
        if len(bucket) >= max_per_minute:
            return False
        bucket.append(now)
        return True


_rate_limiter = SlidingWindowRateLimiter()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    init_db(settings.database_url)
    get_storage()
    get_queue()
    logger.info("service_started", version=__version__, adapter_mode=settings.adapter_mode)
    yield
    logger.info("service_stopping")


def create_app() -> FastAPI:
    app = FastAPI(
        title="IncuBrix Script & Voiceover to B-roll Video Engine",
        version=__version__,
        description=(
            "Production-ready containerized API service that turns voiceovers, scripts, or "
            "IncuBrix scene lists into licence-clean, caption-synced B-roll videos across "
            "16:9, 9:16, and 1:1 aspect ratios."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)

    @app.middleware("http")
    async def observability_auth_ratelimit_middleware(request: Request, call_next):
        corr_id = request.headers.get("X-Correlation-Id") or f"corr_{uuid.uuid4().hex[:12]}"
        token = correlation_id_ctx.set(corr_id)
        settings = get_settings()
        t0 = time.perf_counter()
        try:
            path = request.url.path
            is_protected = path.startswith("/api/v1/") and not path.startswith("/api/v1/files/")

            # Optional Bearer token protection for public live demo
            if is_protected and settings.api_access_token:
                auth_header = request.headers.get("Authorization", "")
                expected = f"Bearer {settings.api_access_token}"
                if auth_header != expected:
                    err = AppError("UNAUTHORIZED", "Valid Bearer token required.", status_code=401)
                    resp = JSONResponse(status_code=401, content=err.to_dict())
                    resp.headers["X-Correlation-Id"] = corr_id
                    return resp

            # Sliding-window rate limiting on write endpoints
            if path.startswith("/api/v1/broll/") and request.method == "POST":
                client_ip = request.client.host if request.client else "unknown"
                if not _rate_limiter.allow(client_ip, settings.rate_limit_per_minute):
                    err = AppError(
                        "RATE_LIMIT_EXCEEDED",
                        f"Rate limit exceeded ({settings.rate_limit_per_minute} POST requests/minute).",
                        status_code=429,
                        retryable=True,
                    )
                    resp = JSONResponse(status_code=429, content=err.to_dict())
                    resp.headers["X-Correlation-Id"] = corr_id
                    return resp

            response = await call_next(request)
            response.headers["X-Correlation-Id"] = corr_id
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
            logger.info(
                "http_request",
                method=request.method,
                path=path,
                status_code=response.status_code,
                latency_ms=elapsed_ms,
            )
            return response
        finally:
            correlation_id_ctx.reset(token)

    app.include_router(v1_router)

    @app.get("/health", response_model=HealthResponse, tags=["Observability"])
    async def health() -> HealthResponse:
        return HealthResponse(status="ok", version=__version__)

    @app.get("/ready", response_model=ReadyResponse, tags=["Observability"])
    async def ready() -> JSONResponse:
        checks = {
            "database": check_db_ready(),
            "storage": get_storage().is_ready(),
            "queue": get_queue().is_ready(),
            "ffmpeg": shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None,
        }
        all_ok = all(checks.values())
        payload = ReadyResponse(status="ready" if all_ok else "degraded", checks=checks)
        return JSONResponse(status_code=200 if all_ok else 503, content=payload.model_dump())

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def demo_ui() -> HTMLResponse:
        ui_path = Path(__file__).parent / "static" / "index.html"
        if ui_path.exists():
            return HTMLResponse(content=ui_path.read_text(encoding="utf-8"))
        return HTMLResponse(content="<h1>IncuBrix B-roll Engine</h1><p>Visit <a href='/docs'>/docs</a>.</p>")

    return app


app = create_app()
