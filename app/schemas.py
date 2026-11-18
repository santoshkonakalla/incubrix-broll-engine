"""Pydantic v2 API & artifact schemas matching the IncuBrix Assessment 3 contract."""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

AspectRatio = Literal["16:9", "9:16", "1:1"]
JobStatus = Literal["queued", "running", "succeeded", "failed"]


class SceneHintInput(BaseModel):
    text: str = Field(..., min_length=1, description="Narration text for this scene.")
    visual_hint: Optional[str] = Field(
        default=None, description="Optional visual direction from IncuBrix."
    )


class CaptionsConfig(BaseModel):
    burn_in: bool = Field(default=True, description="Burn captions directly into rendered MP4.")
    srt: bool = Field(default=True, description="Generate downloadable captions.srt file.")


class CreateBrollJobRequest(BaseModel):
    audio_url: Optional[str] = Field(
        default=None, description="HTTP(S) URL to voiceover audio (mp3, wav, m4a, up to 10 min)."
    )
    script: Optional[str] = Field(
        default=None, description="Plain-text narration script (synthesized via TTS)."
    )
    scenes: Optional[List[SceneHintInput]] = Field(
        default=None, description="Optional IncuBrix scene list with text + visual_hint."
    )
    voice: str = Field(
        default="en-IN-female",
        description="Voice identifier used when input is script/scenes (e.g. en-IN-female, hi-IN-female, ta-IN-female).",
    )
    language: Optional[str] = Field(
        default=None,
        description="Optional ISO language hint (en, hi, ta). Auto-detected from voice/audio if omitted.",
    )
    aspect_ratios: List[AspectRatio] = Field(
        default_factory=lambda: ["16:9", "9:16"],
        min_length=1,
        description="Target output aspect ratios (any combination of 16:9, 9:16, 1:1).",
    )
    style: str = Field(
        default="bright, modern, real-world footage",
        description="Visual style guidance for B-roll selection and generation.",
    )
    avoid: List[str] = Field(
        default_factory=list,
        description="Optional content/concepts to avoid (e.g., people's faces, alcohol).",
    )
    captions: CaptionsConfig = Field(default_factory=CaptionsConfig)
    callback_url: Optional[str] = Field(
        default=None,
        description="Optional webhook URL invoked on completion with HMAC-SHA256 X-IncuBrix-Signature.",
    )
    seed: Optional[int] = Field(
        default=None, description="Optional deterministic seed recorded in manifest.json."
    )
    disabled_sources: Optional[List[str]] = Field(
        default=None,
        description="Optional per-job footage sources to disable (useful for resilience/failover testing).",
    )

    @field_validator("aspect_ratios")
    @classmethod
    def _dedupe_ratios(cls, v: List[AspectRatio]) -> List[AspectRatio]:
        seen: List[AspectRatio] = []
        for item in v:
            if item not in seen:
                seen.append(item)
        return seen

    @model_validator(mode="after")
    def _validate_one_input_mode(self) -> "CreateBrollJobRequest":
        has_audio = bool(self.audio_url and self.audio_url.strip())
        has_script = self.script is not None
        has_scenes = self.scenes is not None

        if not (has_audio or has_script or has_scenes):
            raise ValueError("Provide one of 'audio_url', 'script', or 'scenes'.")
        if has_script and not self.script.strip():
            raise ValueError("EMPTY_SCRIPT: 'script' cannot be empty or whitespace.")
        if has_scenes and (len(self.scenes) == 0 or not any(s.text.strip() for s in self.scenes)):
            raise ValueError("EMPTY_SCRIPT: 'scenes' must contain at least one non-empty scene.")
        return self


class SwapSceneClipRequest(BaseModel):
    query: Optional[str] = Field(
        default=None, description="New search/generation query for replacement clip."
    )
    preferred_source: Optional[str] = Field(
        default=None, description="Optional preferred footage source (pexels, pixabay, generative, statcard)."
    )
    visual_hint: Optional[str] = Field(
        default=None, description="Optional explicit visual direction."
    )
    clip_url: Optional[str] = Field(
        default=None, description="Optional direct licensed clip URL to use."
    )
    licence: Optional[str] = Field(
        default="User-supplied Commercial Licence",
        description="Licence string when clip_url is directly supplied.",
    )
    attribution: Optional[str] = Field(
        default="User-supplied clip",
        description="Attribution string when clip_url is directly supplied.",
    )


class JobAcceptedResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: int = Field(ge=0, le=100)
    status_url: str
    idempotent_replay: bool = False


class ResultLinks(BaseModel):
    videos: Dict[str, str] = Field(default_factory=dict, description="Map of aspect ratio -> download URL.")
    captions_srt: Optional[str] = None
    timeline_json: Optional[str] = None
    manifest_json: Optional[str] = None


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: int = Field(ge=0, le=100)
    stage: str = "queued"
    created_at: datetime
    updated_at: datetime
    result: Optional[ResultLinks] = None
    warnings: List[str] = Field(default_factory=list)
    error: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None


class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float


class ClipLicenceInfo(BaseModel):
    source: str
    clip_id: str
    licence: str
    attribution: str
    source_url: str
    commercial_use_allowed: bool = True
    watermarked: bool = False
    ai_generated: bool = False


class TimelineScene(BaseModel):
    scene_index: int
    start_time: float
    end_time: float
    duration: float
    narration_text: str
    visual_hint: Optional[str] = None
    search_query: str
    disambiguation_notes: Optional[str] = None
    safety_flags: List[str] = Field(default_factory=list)
    clip_id: str
    source: str
    licence: str
    attribution: str
    source_url: str
    ai_generated: bool
    selection_rule: str
    selection_reason: str
    relevance_score: float = Field(ge=1.0, le=5.0)
    vlm_rationale: str = ""
    crop_focus_x: float = Field(default=0.5, ge=0.0, le=1.0)
    color_normalized: bool = True


class TimelineDocument(BaseModel):
    job_id: str
    language: str
    total_duration: float
    aspect_ratios: List[str]
    scene_count: int
    repeat_clip_count: int = 0
    average_relevance_score: float
    scenes: List[TimelineScene]


class ProviderUsageEntry(BaseModel):
    role: str
    provider: str
    model_or_version: str
    calls: int = 0
    estimated_cost_usd: float = 0.0


class ManifestDocument(BaseModel):
    schema_version: str = "1.0.0"
    job_id: str
    created_at: datetime
    completed_at: datetime
    seed: int
    settings: Dict[str, Any]
    models_and_providers: List[ProviderUsageEntry]
    timings_seconds: Dict[str, float]
    cost_estimate_usd: Dict[str, float]
    qa_checks: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    rights_and_licences: List[ClipLicenceInfo] = Field(default_factory=list)
    ai_generated_clip_count: int = 0


class ErrorBody(BaseModel):
    code: str
    message: str
    retryable: bool
    details: Dict[str, Any] = Field(default_factory=dict)


class ErrorEnvelope(BaseModel):
    error: ErrorBody


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str


class ReadyResponse(BaseModel):
    status: Literal["ready", "degraded"]
    checks: Dict[str, bool]
