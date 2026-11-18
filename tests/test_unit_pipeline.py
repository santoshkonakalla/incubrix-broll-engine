"""Unit tests covering safety guardrails, homonyms, pacing, captions, dedup, framing, and HMAC webhooks."""

from pathlib import Path

from app.adapters.mock_adapters import generate_mock_video_clip, segment_words_into_scenes
from app.api.webhooks import sign_payload, verify_signature
from app.pipeline.captions import build_caption_cues, select_caption_font, write_srt
from app.pipeline.dedup import ClipDeduplicator
from app.pipeline.framing import compute_crop_expression, detect_subject_focus_x
from app.pipeline.safety import analyze_scene_safety
from app.schemas import WordTimestamp


def test_homonym_disambiguation_uses_full_script_context():
    script_tech = "Modern cloud computing runs on Linux servers and Python scripts automate daily deployments."
    a_cloud = analyze_scene_safety("Modern cloud computing runs fast.", script_tech, [])
    assert any("server" in r or "data center" in r for r in a_cloud.homonym_resolutions)
    assert "sky" in a_cloud.negative_keywords

    a_py = analyze_scene_safety("Python scripts automate work.", script_tech, [])
    assert any("developer" in r or "code" in r for r in a_py.homonym_resolutions)
    assert "snake" in a_py.negative_keywords

    script_nature = "We sat beside the grassy river bank watching a wild python in the jungle."
    a_bank = analyze_scene_safety("We sat beside the grassy river bank.", script_nature, [])
    assert any("river" in r for r in a_bank.homonym_resolutions)
    assert "money" in a_bank.negative_keywords


def test_safety_blocks_ai_faces_and_brand_logos():
    a_person = analyze_scene_safety("Elon Musk announced a new rocket launch today.", "Elon Musk announced a new rocket.", [])
    assert a_person.block_ai_generation is True
    assert "NAMED_PERSON_NO_AI_FACE" in a_person.flags

    a_brand = analyze_scene_safety("Apple unveiled a new smartphone design.", "Apple unveiled a new smartphone.", [])
    assert a_brand.block_ai_generation is True
    assert "BRAND_NO_INVENTED_LOGO" in a_brand.flags
    assert a_brand.suggested_visual_override is not None


def test_sensitive_news_and_statistics_detection():
    a_sad = analyze_scene_safety("Families mourned after the devastating flood tragedy.", "Flood tragedy news.", [])
    assert a_sad.is_sensitive is True
    assert "SENSITIVE_RESPECTFUL_TONE" in a_sad.flags
    assert "smiling" in a_sad.negative_keywords

    a_stat = analyze_scene_safety("Global sales grew 40% in 2025 across all regions.", "Sales grew 40% in 2025.", [])
    assert a_stat.has_statistics is True
    assert "40%" in (a_stat.extracted_stat or "")


def test_scene_pacing_prevents_flashing_short_clips_and_covers_full_duration():
    # Simulate 20 words across 12 seconds with a very short 0.4s sentence in the middle
    words = [
        WordTimestamp(word="Welcome", start=0.0, end=0.8),
        WordTimestamp(word="everyone", start=0.8, end=1.6),
        WordTimestamp(word="today.", start=1.6, end=2.6),
        WordTimestamp(word="Yes.", start=2.7, end=3.1),  # 0.4s short fragment -> must merge!
        WordTimestamp(word="Building", start=3.2, end=4.5),
        WordTimestamp(word="software", start=4.5, end=6.0),
        WordTimestamp(word="takes", start=6.0, end=7.2),
        WordTimestamp(word="careful", start=7.2, end=8.6),
        WordTimestamp(word="planning", start=8.6, end=10.2),
        WordTimestamp(word="daily.", start=10.2, end=12.0),
    ]
    scenes = segment_words_into_scenes(words, total_duration=12.0, min_scene_duration=2.5, max_scene_duration=9.0)
    assert scenes[0][0] == 0.0
    assert scenes[-1][1] == 12.0
    for idx, (s, e, _txt) in enumerate(scenes):
        assert (e - s) >= 2.5, f"Flashing clip detected: {e - s}s"
        if idx > 0:
            assert abs(s - scenes[idx - 1][1]) < 0.01


def test_multilingual_captions_and_fonts(tmp_path: Path):
    words = [
        WordTimestamp(word="नमस्ते", start=0.0, end=0.6),
        WordTimestamp(word="दुनिया।", start=0.6, end=1.3),
    ]
    cues = build_caption_cues(words)
    srt_path = write_srt(cues, tmp_path / "hi.srt")
    content = srt_path.read_text(encoding="utf-8")
    assert "00:00:00,000 -->" in content
    assert "नमस्ते दुनिया।" in content
    assert select_caption_font("hi") == "Noto Sans Devanagari"
    assert select_caption_font("ta") == "Noto Sans Tamil"
    assert select_caption_font("en") == "Noto Sans"


def test_edge_subject_smart_crop_9x16(tmp_path: Path):
    # Generate a 16:9 clip where the bright subject sits near the far right edge (x = 0.85)
    clip = generate_mock_video_clip(tmp_path / "edge_right.mp4", label="edge subject", duration=2.5, edge_subject_x=0.85)
    focus_x = detect_subject_focus_x(clip)
    assert focus_x > 0.65, f"Expected subject detection toward right edge, got {focus_x}"

    crop_w, crop_h, crop_x, crop_y = compute_crop_expression(1280, 720, 1080, 1920, focus_x=focus_x)
    assert crop_h == 720
    assert crop_x > 400, f"Smart crop should shift right to preserve edge subject, got crop_x={crop_x}"
    assert crop_x + crop_w <= 1280


def test_clip_deduplicator_catches_id_and_visual_duplicates(tmp_path: Path):
    c1 = generate_mock_video_clip(tmp_path / "c1.mp4", label="same clip", duration=2.0, seed=10)
    c2 = generate_mock_video_clip(tmp_path / "c2.mp4", label="same clip", duration=2.0, seed=10)
    c3 = generate_mock_video_clip(tmp_path / "c3.mp4", label="different clip", duration=2.0, seed=99)

    dedup = ClipDeduplicator()
    dedup.register("pexels_1", "https://pexels.com/1", c1)
    assert dedup.is_duplicate_metadata("pexels_1", "https://pexels.com/other") is True
    assert dedup.is_duplicate_visual(c2) is True
    assert dedup.is_duplicate_visual(c3) is False


def test_webhook_hmac_signature():
    payload = b'{"job_id":"job_123","status":"succeeded"}'
    sig = sign_payload(payload, "my-secret")
    assert sig.startswith("sha256=")
    assert verify_signature(payload, sig, "my-secret") is True
    assert verify_signature(payload, sig, "wrong-secret") is False
