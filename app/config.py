"""Application configuration loaded strictly from environment variables / .env."""

from functools import lru_cache
import os
from pathlib import Path
import sys
from typing import List, Literal

if sys.platform == "win32":
    _gyan_path = r"C:\Users\santo\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-essentials_build\bin"
    _cur_items = [p for p in os.environ.get("PATH", "").split(";") if p.strip()]
    # Ensure working Gyan FFmpeg takes highest priority over broken conda Library\bin\ffmpeg.exe
    _filtered = [p for p in _cur_items if "incubrix\\library\\bin" not in p.lower()]
    if Path(_gyan_path).exists():
        os.environ["PATH"] = _gyan_path + ";" + ";".join(_filtered)
    else:
        os.environ["PATH"] = ";".join(_filtered)

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Service & Security
    app_env: str = Field(default="development")
    app_host: str = Field(default="0.0.0.0")
    app_port: int = Field(default=8000)
    log_level: str = Field(default="INFO")
    api_access_token: str = Field(default="")
    rate_limit_per_minute: int = Field(default=30, ge=1)
    webhook_hmac_secret: str = Field(default="incubrix-dev-webhook-secret")
    public_base_url: str = Field(default="http://localhost:8000")

    # Adapter mode & provider selection
    adapter_mode: Literal["mock", "live"] = Field(default="mock")
    asr_provider: str = Field(default="mock")
    tts_provider: str = Field(default="mock")
    planner_provider: str = Field(default="mock")
    vlm_scorer_provider: str = Field(default="mock")
    footage_sources: str = Field(default="pexels,pixabay,generative,statcard")
    disabled_sources: str = Field(default="")

    # External API keys (never logged or serialized into manifests)
    gemini_api_key: str = Field(default="")
    groq_api_key: str = Field(default="")
    pexels_api_key: str = Field(default="")
    pixabay_api_key: str = Field(default="")

    # Model names
    gemini_planner_model: str = Field(default="gemini-3.5-flash-lite")
    gemini_tts_model: str = Field(default="gemini-2.5-flash-preview-tts")
    gemini_image_model: str = Field(default="gemini-3.1-flash-lite-image")
    groq_asr_model: str = Field(default="whisper-large-v3-turbo")
    groq_vlm_model: str = Field(default="qwen/qwen3.8-27b")

    # Pipeline limits & quality parameters
    max_audio_duration_seconds: float = Field(default=600.0, gt=0)
    max_upload_mb: int = Field(default=100, gt=0)
    min_scene_duration_seconds: float = Field(default=2.5, gt=0.5)
    max_scene_duration_seconds: float = Field(default=9.0, gt=2.0)
    stock_min_relevance_score: float = Field(default=3.5, ge=1.0, le=5.0)
    target_fps: int = Field(default=30, ge=24, le=60)
    render_resolution: Literal["720p", "1080p"] = Field(default="720p", alias="RENDER_RESOLUTION")
    color_normalize_enabled: bool = Field(default=True)
    default_seed: int = Field(default=42)
    worker_concurrency: int = Field(default=2, ge=1, le=16)
    job_timeout_seconds: int = Field(default=600, ge=30)

    # Infrastructure
    database_url: str = Field(default="sqlite:///./.local_storage/incubrix.db")
    redis_url: str = Field(default="")
    queue_backend: Literal["inline", "redis"] = Field(default="inline")
    storage_backend: Literal["local", "s3"] = Field(default="local")
    local_storage_dir: Path = Field(default=Path("./.local_storage"))
    temp_work_dir: Path = Field(default=Path("./.tmp_jobs"))

    # S3 / MinIO
    s3_endpoint_url: str = Field(default="http://minio:9000")
    s3_public_endpoint_url: str = Field(default="http://localhost:9000")
    s3_access_key_id: str = Field(default="minioadmin")
    s3_secret_access_key: str = Field(default="minioadmin")
    s3_bucket_name: str = Field(default="incubrix-broll")
    s3_region: str = Field(default="us-east-1")

    @field_validator("local_storage_dir", "temp_work_dir", mode="before")
    @classmethod
    def _coerce_path(cls, v: str | Path) -> Path:
        return Path(v)

    @property
    def enabled_footage_sources(self) -> List[str]:
        disabled = {s.strip().lower() for s in self.disabled_sources.split(",") if s.strip()}
        return [
            s.strip().lower()
            for s in self.footage_sources.split(",")
            if s.strip() and s.strip().lower() not in disabled
        ]

    def effective_provider(self, explicit: str, live_default: str) -> str:
        if self.adapter_mode == "mock":
            return "mock"
        return explicit if explicit != "mock" else live_default


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.local_storage_dir.mkdir(parents=True, exist_ok=True)
    settings.temp_work_dir.mkdir(parents=True, exist_ok=True)
    return settings
