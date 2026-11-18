"""Standardized JSON error envelope and UPPER_SNAKE_CASE error codes (Integration Standard #5)."""

from typing import Any, Dict, Optional

from fastapi import HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


ERROR_CODES: Dict[str, str] = {
    "INVALID_INPUT": "Request payload or parameters failed validation.",
    "EMPTY_SCRIPT": "Provided script or scenes list contains no speakable narration text.",
    "UNSUPPORTED_MEDIA_TYPE": "Uploaded or referenced audio file format is not supported (allowed: mp3, wav, m4a).",
    "AUDIO_TOO_LONG": "Audio exceeds the maximum supported duration (10 minutes).",
    "AUDIO_SILENT_OR_NO_SPEECH": "Audio contains only silence or non-speech audio (e.g., music only) with no detectable narration.",
    "AUDIO_DOWNLOAD_FAILED": "Failed to fetch the audio file from the provided audio_url.",
    "AUDIO_CORRUPTED": "Audio file could not be decoded by ffprobe/ffmpeg.",
    "JOB_NOT_FOUND": "No job exists with the requested job_id.",
    "SCENE_NOT_FOUND": "Requested scene_index is out of bounds for this job.",
    "JOB_NOT_READY": "Job must be in 'succeeded' state before swapping a scene clip.",
    "IDEMPOTENCY_CONFLICT": "Idempotency-Key was previously used with a different request payload.",
    "UNAUTHORIZED": "Missing or invalid bearer access token.",
    "RATE_LIMIT_EXCEEDED": "Too many requests from this client. Retry after cooldown.",
    "PROVIDER_UNAVAILABLE": "An upstream AI or media provider is temporarily unreachable.",
    "RENDER_FAILED": "Video normalization or assembly failed during FFmpeg processing.",
    "STORAGE_ERROR": "Failed to read or write artifact in object storage.",
    "INTERNAL_ERROR": "An unexpected server error occurred.",
}


class AppError(Exception):
    """Domain error serialized to Integration Standard #5 JSON format."""

    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 400,
        retryable: bool = False,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code.upper()
        self.message = message
        self.status_code = status_code
        self.retryable = retryable
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "retryable": self.retryable,
                "details": self.details,
            }
        }


async def app_error_handler(_request: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=exc.to_dict())


async def validation_error_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    safe_errors = jsonable_encoder(
        exc.errors(),
        custom_encoder={Exception: lambda e: str(e)},
    )
    code = "INVALID_INPUT"
    for item in safe_errors:
        if "EMPTY_SCRIPT" in str(item.get("msg", "")):
            code = "EMPTY_SCRIPT"
            break
    err = AppError(
        code=code,
        message="Request validation failed.",
        status_code=422,
        retryable=False,
        details={"validation_errors": safe_errors},
    )
    return JSONResponse(status_code=422, content=err.to_dict())


async def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
    code = "UNAUTHORIZED" if exc.status_code == 401 else "INVALID_INPUT"
    if exc.status_code == 404:
        code = "JOB_NOT_FOUND"
    elif exc.status_code == 429:
        code = "RATE_LIMIT_EXCEEDED"
    err = AppError(
        code=code,
        message=str(exc.detail),
        status_code=exc.status_code,
        retryable=exc.status_code in (429, 502, 503, 504),
    )
    return JSONResponse(status_code=exc.status_code, content=err.to_dict())


async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    err = AppError(
        code="INTERNAL_ERROR",
        message="An unexpected internal error occurred.",
        status_code=500,
        retryable=True,
        details={"exception_type": type(exc).__name__},
    )
    return JSONResponse(status_code=500, content=err.to_dict())
