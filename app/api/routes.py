"""Versioned /api/v1 HTTP endpoints (Integration Standard #2, #3, #4, #5, #7)."""

import hashlib
import json
import mimetypes
from pathlib import Path
from typing import Any, Dict, Optional
import uuid

from fastapi import APIRouter, File, Form, Header, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from app.config import get_settings
from app.db import IdempotencyRecord, JobRecord, get_session
from app.errors import AppError
from app.logging_config import correlation_id_ctx, get_logger
from app.pipeline.orchestrator import execute_swap_scene_clip
from app.queue import get_queue
from app.schemas import (
    CreateBrollJobRequest,
    ErrorEnvelope,
    JobAcceptedResponse,
    JobStatusResponse,
    ResultLinks,
    SwapSceneClipRequest,
)
from app.storage import get_storage

logger = get_logger("api")
router = APIRouter(prefix="/api/v1", tags=["B-roll Engine v1"])


def _canonical_hash(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _base_url_from_request(request: Request) -> str:
    settings = get_settings()
    if settings.public_base_url and "localhost" not in settings.public_base_url:
        return settings.public_base_url.rstrip("/")
    return str(request.base_url).rstrip("/")


def _create_or_replay_job(
    *,
    endpoint: str,
    payload_dict: Dict[str, Any],
    idempotency_key: Optional[str],
    base_url: str,
) -> JobAcceptedResponse:
    req_hash = _canonical_hash(payload_dict)

    with get_session() as db:
        if idempotency_key:
            existing = db.get(IdempotencyRecord, f"{endpoint}:{idempotency_key}")
            if existing:
                if existing.request_hash != req_hash:
                    raise AppError(
                        "IDEMPOTENCY_CONFLICT",
                        "Idempotency-Key was already used with a different request payload.",
                        status_code=409,
                        retryable=False,
                    )
                existing_job = db.get(JobRecord, existing.job_id)
                return JobAcceptedResponse(
                    job_id=existing.job_id,
                    status=existing_job.status if existing_job else "queued",
                    progress=existing_job.progress if existing_job else 0,
                    status_url=f"{base_url}/api/v1/jobs/{existing.job_id}",
                    idempotent_replay=True,
                )

        job_id = f"job_{uuid.uuid4().hex[:12]}"
        job = JobRecord(
            job_id=job_id,
            status="queued",
            progress=0,
            stage="queued",
            request_payload=payload_dict,
            warnings=[],
            correlation_id=correlation_id_ctx.get(),
        )
        db.add(job)
        if idempotency_key:
            db.add(
                IdempotencyRecord(
                    idempotency_key=f"{endpoint}:{idempotency_key}",
                    endpoint=endpoint,
                    request_hash=req_hash,
                    job_id=job_id,
                )
            )
        db.commit()

    get_queue().enqueue(job_id)
    logger.info("job_accepted", job_id=job_id, idempotency_key=bool(idempotency_key))
    return JobAcceptedResponse(
        job_id=job_id,
        status="queued",
        progress=0,
        status_url=f"{base_url}/api/v1/jobs/{job_id}",
        idempotent_replay=False,
    )


@router.post(
    "/broll/jobs",
    response_model=JobAcceptedResponse,
    status_code=202,
    responses={400: {"model": ErrorEnvelope}, 409: {"model": ErrorEnvelope}, 422: {"model": ErrorEnvelope}},
    summary="Submit a B-roll generation job from audio_url, plain script, or IncuBrix scenes",
)
async def create_broll_job(
    request: Request,
    body: CreateBrollJobRequest,
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
) -> JSONResponse:
    base_url = _base_url_from_request(request)
    accepted = _create_or_replay_job(
        endpoint="POST:/api/v1/broll/jobs",
        payload_dict=body.model_dump(mode="json"),
        idempotency_key=idempotency_key,
        base_url=base_url,
    )
    return JSONResponse(status_code=202, content=accepted.model_dump(mode="json"))


@router.post(
    "/broll/jobs/upload",
    response_model=JobAcceptedResponse,
    status_code=202,
    responses={400: {"model": ErrorEnvelope}, 409: {"model": ErrorEnvelope}, 415: {"model": ErrorEnvelope}},
    summary="Submit a B-roll generation job with a direct audio file upload (mp3, wav, m4a)",
)
async def create_broll_job_upload(
    request: Request,
    audio_file: UploadFile = File(..., description="Voiceover audio file (mp3, wav, or m4a, up to 10 min)."),
    metadata: str = Form(
        default='{"aspect_ratios": ["16:9", "9:16"]}',
        description="JSON string matching CreateBrollJobRequest options (aspect_ratios, style, avoid, captions, callback_url).",
    ),
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
) -> JSONResponse:
    settings = get_settings()
    try:
        meta_dict = json.loads(metadata) if metadata.strip() else {}
    except json.JSONDecodeError as exc:
        raise AppError("INVALID_INPUT", f"metadata must be valid JSON: {exc}", status_code=400) from exc

    data = await audio_file.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(data) == 0:
        raise AppError("AUDIO_CORRUPTED", "Uploaded audio_file is empty.", status_code=400)
    if len(data) > max_bytes:
        raise AppError("INVALID_INPUT", f"Uploaded file exceeds {settings.max_upload_mb} MB limit.", status_code=413)

    ext = Path(audio_file.filename or "upload.wav").suffix.lower() or ".wav"
    if ext not in {".mp3", ".wav", ".m4a", ".mp4", ".aac"}:
        raise AppError(
            "UNSUPPORTED_MEDIA_TYPE",
            f"Unsupported file extension '{ext}'. Allowed: .mp3, .wav, .m4a.",
            status_code=415,
            details={"filename": audio_file.filename},
        )

    file_sha = hashlib.sha256(data).hexdigest()[:16]
    upload_key = f"uploads/{file_sha}_{Path(audio_file.filename or 'audio' + ext).name}"
    get_storage().put_bytes(data, upload_key, content_type=audio_file.content_type or "application/octet-stream")

    # Validate options via CreateBrollJobRequest using synthetic audio_url placeholder
    meta_dict["audio_url"] = f"uploaded://{upload_key}"
    validated = CreateBrollJobRequest.model_validate(meta_dict)
    payload_dict = validated.model_dump(mode="json")
    payload_dict["_uploaded_audio_key"] = upload_key
    payload_dict["_upload_sha256"] = file_sha

    base_url = _base_url_from_request(request)
    accepted = _create_or_replay_job(
        endpoint="POST:/api/v1/broll/jobs/upload",
        payload_dict=payload_dict,
        idempotency_key=idempotency_key,
        base_url=base_url,
    )
    return JSONResponse(status_code=202, content=accepted.model_dump(mode="json"))


@router.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    responses={404: {"model": ErrorEnvelope}},
    summary="Poll background job status, progress (0-100), warnings, and artifact download links",
)
async def get_job_status(request: Request, job_id: str) -> JobStatusResponse:
    with get_session() as db:
        rec = db.get(JobRecord, job_id)
        if not rec:
            raise AppError("JOB_NOT_FOUND", f"No job found with id '{job_id}'.", status_code=404)

    result_links: Optional[ResultLinks] = None
    if rec.result_keys:
        base_url = _base_url_from_request(request)
        store = get_storage()
        rk = rec.result_keys
        result_links = ResultLinks(
            videos={ratio: store.build_download_url(key, base_url) for ratio, key in (rk.get("videos") or {}).items()},
            captions_srt=store.build_download_url(rk["captions_srt"], base_url) if rk.get("captions_srt") else None,
            timeline_json=store.build_download_url(rk["timeline_json"], base_url) if rk.get("timeline_json") else None,
            manifest_json=store.build_download_url(rk["manifest_json"], base_url) if rk.get("manifest_json") else None,
        )

    return JobStatusResponse(
        job_id=rec.job_id,
        status=rec.status,
        progress=rec.progress,
        stage=rec.stage,
        created_at=rec.created_at,
        updated_at=rec.updated_at,
        result=result_links,
        warnings=list(rec.warnings or []),
        error=rec.error_payload,
        metrics=rec.metrics,
    )


@router.post(
    "/jobs/{job_id}/cancel",
    summary="Cancel a queued or running job and clear worker queue",
)
async def cancel_job(job_id: str) -> JSONResponse:
    with get_session() as db:
        rec = db.get(JobRecord, job_id)
        if not rec:
            raise AppError("JOB_NOT_FOUND", f"No job found with id '{job_id}'.", status_code=404)
        if rec.status in ("succeeded", "failed"):
            return JSONResponse(status_code=200, content={"status": rec.status, "message": "Job already finished."})
        rec.status = "failed"
        rec.stage = "cancelled"
        rec.error_payload = {"code": "CANCELLED_BY_USER", "message": "Job was cancelled by user."}
        db.add(rec)
        db.commit()
    logger.info("job_cancelled_by_user", job_id=job_id)
    return JSONResponse(status_code=200, content={"status": "cancelled", "message": f"Job {job_id} cancelled."})


@router.post(
    "/broll/jobs/{job_id}/scenes/{scene_index}/swap",
    responses={404: {"model": ErrorEnvelope}, 409: {"model": ErrorEnvelope}},
    summary="Replace a single scene's clip and fast re-render only the affected segment",
)
async def swap_scene_clip(
    job_id: str,
    scene_index: int,
    body: SwapSceneClipRequest,
) -> Dict[str, Any]:
    return execute_swap_scene_clip(job_id=job_id, scene_index=scene_index, swap_req=body)


@router.get(
    "/files/{object_key:path}",
    summary="Download a generated artifact (MP4, SRT, timeline.json, manifest.json) from object storage",
)
async def download_artifact(object_key: str) -> Response:
    if ".." in object_key:
        raise AppError("INVALID_INPUT", "Invalid object path.", status_code=400)
    data = get_storage().get_bytes(object_key)
    guessed, _ = mimetypes.guess_type(object_key)
    media_type = guessed or "application/octet-stream"
    if object_key.endswith(".srt"):
        media_type = "text/plain; charset=utf-8"
    elif object_key.endswith(".json"):
        media_type = "application/json; charset=utf-8"
    elif object_key.endswith(".mp4"):
        media_type = "video/mp4"
    return Response(content=data, media_type=media_type)
