"""Audio ingestion, magic-byte/ffprobe validation, duration & silence checks (Must-Have #1 & Hard Case #6)."""

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Tuple

import httpx

from app.errors import AppError

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".mp4", ".aac"}
ALLOWED_CODECS = {"mp3", "pcm_s16le", "pcm_s24le", "pcm_f32le", "pcm_u8", "aac", "alac", "mp2", "flac"}


@dataclass
class AudioProbeResult:
    duration_seconds: float
    codec_name: str
    sample_rate: int
    channels: int
    mean_volume_db: float
    max_volume_db: float


def _check_magic_bytes(file_path: Path) -> bool:
    header = file_path.read_bytes()[:16]
    if len(header) < 4:
        return False
    # WAV: RIFF....WAVE
    if header[:4] == b"RIFF" and header[8:12] == b"WAVE":
        return True
    # MP3: ID3 tag or MPEG frame sync (0xFF 0xFB / 0xF3 / 0xF2 / 0xFA / 0xE0+)
    if header[:3] == b"ID3" or (header[0] == 0xFF and (header[1] & 0xE0) == 0xE0):
        return True
    # M4A / MP4 ISO Base Media: ....ftyp
    if header[4:8] == b"ftyp":
        return True
    return False


def download_audio_url(url: str, dest_dir: Path, max_mb: int = 100) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    if not (url.startswith("http://") or url.startswith("https://")):
        raise AppError(
            "INVALID_INPUT",
            "audio_url must start with http:// or https://.",
            status_code=400,
            retryable=False,
        )
    ext = Path(url.split("?")[0]).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        ext = ".audio"
    dest_path = dest_dir / f"input_audio{ext}"
    max_bytes = max_mb * 1024 * 1024

    try:
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            with client.stream("GET", url) as resp:
                if resp.status_code >= 400:
                    raise AppError(
                        "AUDIO_DOWNLOAD_FAILED",
                        f"Failed to download audio_url (HTTP {resp.status_code}).",
                        status_code=400,
                        retryable=resp.status_code >= 500,
                        details={"http_status": resp.status_code},
                    )
                total = 0
                with dest_path.open("wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=65536):
                        total += len(chunk)
                        if total > max_bytes:
                            raise AppError(
                                "INVALID_INPUT",
                                f"Audio exceeds maximum allowed size of {max_mb} MB.",
                                status_code=413,
                                retryable=False,
                            )
                        f.write(chunk)
    except AppError:
        raise
    except Exception as exc:
        raise AppError(
            "AUDIO_DOWNLOAD_FAILED",
            f"Could not fetch audio_url: {exc}",
            status_code=400,
            retryable=True,
        ) from exc

    return dest_path


def probe_and_validate_audio(
    file_path: Path,
    max_duration_seconds: float = 600.0,
) -> AudioProbeResult:
    if not file_path.exists() or file_path.stat().st_size < 64:
        raise AppError(
            "AUDIO_CORRUPTED",
            "Audio file is missing or empty.",
            status_code=400,
            retryable=False,
        )

    if not _check_magic_bytes(file_path):
        raise AppError(
            "UNSUPPORTED_MEDIA_TYPE",
            "File does not appear to be a valid mp3, wav, or m4a audio file.",
            status_code=415,
            retryable=False,
            details={"filename": file_path.name},
        )

    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_name,sample_rate,channels,duration:format=duration",
        "-of", "json",
        str(file_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
        raise AppError(
            "AUDIO_CORRUPTED",
            "ffprobe failed to decode audio file.",
            status_code=400,
            retryable=False,
            details={"stderr": proc.stderr[:300]},
        )

    info = json.loads(proc.stdout or "{}")
    streams = info.get("streams") or []
    if not streams:
        raise AppError(
            "UNSUPPORTED_MEDIA_TYPE",
            "Uploaded file contains no audio stream.",
            status_code=415,
            retryable=False,
        )

    stream = streams[0]
    codec = str(stream.get("codec_name", "")).lower()
    if codec and codec not in ALLOWED_CODECS:
        raise AppError(
            "UNSUPPORTED_MEDIA_TYPE",
            f"Unsupported audio codec '{codec}'. Supported formats: mp3, wav, m4a.",
            status_code=415,
            retryable=False,
            details={"codec": codec},
        )

    dur_raw = stream.get("duration") or (info.get("format") or {}).get("duration")
    try:
        duration = float(dur_raw)
    except (TypeError, ValueError):
        raise AppError("AUDIO_CORRUPTED", "Could not determine audio duration.", status_code=400)

    if duration <= 0.3:
        raise AppError("AUDIO_SILENT_OR_NO_SPEECH", "Audio duration is too short (<0.3s).", status_code=400)

    if duration > max_duration_seconds:
        raise AppError(
            "AUDIO_TOO_LONG",
            f"Audio duration ({duration:.1f}s) exceeds maximum allowed ({max_duration_seconds:.0f}s).",
            status_code=400,
            retryable=False,
            details={"duration_seconds": round(duration, 2), "limit_seconds": max_duration_seconds},
        )

    mean_db, max_db = _measure_volume_db(file_path)
    if max_db <= -55.0 or mean_db <= -68.0:
        raise AppError(
            "AUDIO_SILENT_OR_NO_SPEECH",
            "Input audio is effectively silent (no audible voiceover detected).",
            status_code=422,
            retryable=False,
            details={"mean_volume_db": mean_db, "max_volume_db": max_db},
        )

    return AudioProbeResult(
        duration_seconds=duration,
        codec_name=codec,
        sample_rate=int(stream.get("sample_rate") or 24000),
        channels=int(stream.get("channels") or 1),
        mean_volume_db=mean_db,
        max_volume_db=max_db,
    )


def _measure_volume_db(file_path: Path) -> Tuple[float, float]:
    cmd = [
        "ffmpeg",
        "-threads", "2",
        "-i", str(file_path),
        "-af", "volumedetect",
        "-f", "null",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    mean_db = -20.0
    max_db = -5.0
    for line in proc.stderr.splitlines():
        if "mean_volume:" in line:
            try:
                mean_db = float(line.split("mean_volume:")[1].split("dB")[0].strip())
            except ValueError:
                pass
        elif "max_volume:" in line:
            try:
                max_db = float(line.split("max_volume:")[1].split("dB")[0].strip())
            except ValueError:
                pass
    return mean_db, max_db
