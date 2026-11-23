"""Abstract adapter contracts for ASR, TTS, Scene Planning, Footage Sources, and Independent VLM Scoring."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from app.schemas import SceneHintInput, WordTimestamp


@dataclass
class ASRResult:
    text: str
    language: str
    duration_seconds: float
    words: List[WordTimestamp]
    provider: str
    model: str
    estimated_cost_usd: float = 0.0
    speech_detected: bool = True


@dataclass
class TTSResult:
    audio_path: Path
    duration_seconds: float
    language: str
    provider: str
    model: str
    estimated_cost_usd: float = 0.0


@dataclass
class PlannedScene:
    scene_index: int
    start_time: float
    end_time: float
    narration_text: str
    visual_hint: Optional[str]
    search_query: str
    alt_queries: List[str] = field(default_factory=list)
    generative_prompt: str = ""
    disambiguation_notes: Optional[str] = None
    prefer_generative: bool = False
    prefer_statcard: bool = False
    stat_headline: Optional[str] = None
    safety_flags: List[str] = field(default_factory=list)
    negative_keywords: List[str] = field(default_factory=list)
    block_ai_generation: bool = False
    warnings: List[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.1, self.end_time - self.start_time)


@dataclass
class FootageCandidate:
    clip_id: str
    source: str
    download_url: str
    source_url: str
    licence: str
    attribution: str
    duration: float
    width: int
    height: int
    tags: List[str] = field(default_factory=list)
    description: str = ""
    ai_generated: bool = False
    watermarked: bool = False
    commercial_use_allowed: bool = True
    estimated_cost_usd: float = 0.0
    local_path: Optional[Path] = None


@dataclass
class VLMScoreResult:
    relevance_score: float  # 1.0 to 5.0
    rationale: str
    violates_avoid: bool = False
    avoid_reason: Optional[str] = None
    subject_focus_x: Optional[float] = None  # 0.0 (left) to 1.0 (right)
    provider: str = "mock"
    model: str = "mock-vlm-v1"
    estimated_cost_usd: float = 0.0


class ASRAdapter(ABC):
    provider_name: str
    model_name: str

    @abstractmethod
    def transcribe(self, audio_path: Path, language_hint: Optional[str] = None) -> ASRResult:
        raise NotImplementedError


class TTSAdapter(ABC):
    provider_name: str
    model_name: str

    @abstractmethod
    def synthesize(
        self,
        text: str,
        voice: str,
        out_path: Path,
        language_hint: Optional[str] = None,
    ) -> TTSResult:
        raise NotImplementedError


class ScenePlannerAdapter(ABC):
    provider_name: str
    model_name: str

    @abstractmethod
    def plan_scenes(
        self,
        transcript: str,
        words: List[WordTimestamp],
        total_duration: float,
        style: str,
        avoid: List[str],
        explicit_scenes: Optional[List[SceneHintInput]] = None,
        min_scene_duration: float = 2.5,
        max_scene_duration: float = 9.0,
    ) -> List[PlannedScene]:
        raise NotImplementedError


class FootageSourceAdapter(ABC):
    source_name: str
    is_generative: bool = False

    @abstractmethod
    def search(
        self,
        query: str,
        *,
        duration_needed: float,
        orientation: str,
        negative_keywords: List[str],
        limit: int = 4,
        generative_prompt: str = "",
        stat_headline: Optional[str] = None,
        scene_text: str = "",
        work_dir: Optional[Path] = None,
        seed: int = 42,
    ) -> List[FootageCandidate]:
        raise NotImplementedError

    @abstractmethod
    def fetch_clip(self, candidate: FootageCandidate, dest_path: Path) -> Path:
        raise NotImplementedError


class VLMScorerAdapter(ABC):
    provider_name: str
    model_name: str

    @abstractmethod
    def score_clip(
        self,
        video_path: Path,
        narration_text: str,
        visual_intent: str,
        avoid_terms: List[str],
        negative_keywords: List[str],
        candidate_metadata: str = "",
    ) -> VLMScoreResult:
        raise NotImplementedError
