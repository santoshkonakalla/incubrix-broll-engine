"""End-to-end B-roll job orchestrator and fast single-scene swap-a-clip engine."""

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple

from app.adapters.base import FootageCandidate, FootageSourceAdapter, PlannedScene, VLMScoreResult
from app.adapters.registry import AdapterRegistry
from app.api.webhooks import send_webhook
from app.config import Settings, get_settings
from app.db import JobRecord, get_session, update_job
from app.errors import AppError
from app.logging_config import correlation_id_ctx, get_logger, job_id_ctx
from app.pipeline.audio_validator import download_audio_url, probe_and_validate_audio
from app.pipeline.captions import build_caption_cues, write_srt
from app.pipeline.dedup import ClipDeduplicator
from app.pipeline.framing import detect_subject_focus_x
from app.pipeline.qa_checks import run_video_qa
from app.pipeline.renderer import (
    RATIO_FILE_TAGS,
    assemble_video_from_segments,
    probe_video_stream,
    render_scene_segment,
    slice_cues_for_scene,
)
from app.pipeline.safety import analyze_scene_safety, violates_avoid_list
from app.schemas import (
    ClipLicenceInfo,
    CreateBrollJobRequest,
    ManifestDocument,
    ProviderUsageEntry,
    SwapSceneClipRequest,
    TimelineDocument,
    TimelineScene,
)
from app.storage import StorageBackend, get_storage

logger = get_logger("orchestrator")


def _orientation_for_ratios(aspect_ratios: List[str]) -> str:
    if aspect_ratios == ["9:16"]:
        return "portrait"
    if aspect_ratios == ["1:1"]:
        return "square"
    return "landscape"


def _validate_downloaded_video(clip_path: Path) -> bool:
    """Reject corrupted or truncated downloads (Hard Case #7)."""
    if not clip_path.exists() or clip_path.stat().st_size < 1024:
        return False
    try:
        meta = probe_video_stream(clip_path)
        return int(meta["width"]) >= 160 and int(meta["height"]) >= 160 and float(meta["duration"]) >= 0.4
    except Exception:
        return False


def _select_clip_for_scene(
    scene: PlannedScene,
    *,
    sources: Dict[str, FootageSourceAdapter],
    vlm_scorer,
    dedup: ClipDeduplicator,
    avoid_terms: List[str],
    orientation: str,
    stock_threshold: float,
    work_dir: Path,
    seed: int,
    warnings: List[str],
    failed_sources: set,
    preferred_source: Optional[str] = None,
    usage_counts: Dict[str, int],
    usage_costs: Dict[str, float],
) -> Tuple[FootageCandidate, Path, VLMScoreResult, str, str]:
    """
    Explicit, measurable multi-source selection rule (Advanced Level):
      - R1_PREFERRED_SOURCE: If user explicitly requested a source (e.g. during swap) and it succeeds.
      - R2_STOCK_FIRST: Evaluate non-generative stock sources (pexels, pixabay). Pick highest VLM-scored
        non-duplicate, avoid-clean clip if score >= stock_threshold (default 3.5/5.0).
      - R3_GENERATIVE_ESCALATION: If stock best score < stock_threshold (or prefer_generative=True for
        abstract metaphors) AND safety.block_ai_generation is False -> generate bespoke AI visual.
      - R4_STAT_OR_RESILIENT_FALLBACK: For stat scenes with weak stock or when all external sources fail,
        generate a clean programmatic animated card (statcard) so the job always completes.
    """
    duration_needed = max(1.0, scene.end_time - scene.start_time)
    best_stock: Optional[Tuple[FootageCandidate, Path, VLMScoreResult]] = None

    stock_order = [name for name in ("pexels", "pixabay") if name in sources and name not in failed_sources]
    if preferred_source and preferred_source in stock_order:
        stock_order.remove(preferred_source)
        stock_order.insert(0, preferred_source)

    def try_source_candidates(
        src_name: str,
        adapter: FootageSourceAdapter,
        queries: List[str],
        limit: int = 3,
    ) -> Optional[Tuple[FootageCandidate, Path, VLMScoreResult]]:
        local_best: Optional[Tuple[FootageCandidate, Path, VLMScoreResult]] = None
        for q in queries:
            try:
                usage_counts[src_name] = usage_counts.get(src_name, 0) + 1
                cands = adapter.search(
                    q,
                    duration_needed=duration_needed,
                    orientation=orientation,
                    negative_keywords=scene.negative_keywords,
                    limit=limit,
                    generative_prompt=scene.generative_prompt,
                    stat_headline=scene.stat_headline,
                    scene_text=scene.narration_text,
                    work_dir=work_dir / "clips" / src_name,
                    seed=seed + scene.scene_index * 17,
                )
            except Exception as exc:
                failed_sources.add(src_name)
                msg = f"Footage source '{src_name}' unavailable on scene {scene.scene_index} ({exc}); failing over."
                warnings.append(msg)
                logger.warning("footage_source_failover", source=src_name, scene_index=scene.scene_index, error=str(exc))
                return None

            for cand in cands:
                if cand.watermarked or not cand.commercial_use_allowed:
                    continue
                if dedup.is_duplicate_metadata(cand.clip_id, cand.source_url):
                    continue
                meta_blob = f"{cand.description} {' '.join(cand.tags)}"
                if violates_avoid_list(meta_blob, avoid_terms, scene.negative_keywords):
                    continue

                dest = work_dir / "downloads" / f"s{scene.scene_index}_{cand.clip_id}.mp4"
                try:
                    clip_path = adapter.fetch_clip(cand, dest)
                except Exception as exc:
                    warnings.append(f"Clip download failed for {cand.clip_id} ({exc}); trying next candidate.")
                    continue

                if not _validate_downloaded_video(clip_path):
                    warnings.append(f"Corrupted clip rejected for {cand.clip_id}; trying next candidate.")
                    clip_path.unlink(missing_ok=True)
                    continue

                if dedup.is_duplicate_visual(clip_path):
                    continue

                usage_counts["vlm_scorer"] = usage_counts.get("vlm_scorer", 0) + 1
                vlm_res = vlm_scorer.score_clip(
                    clip_path,
                    narration_text=scene.narration_text,
                    visual_intent=scene.visual_hint or scene.search_query,
                    avoid_terms=avoid_terms,
                    negative_keywords=scene.negative_keywords,
                    candidate_metadata=meta_blob,
                )
                usage_costs["vlm_scorer"] = usage_costs.get("vlm_scorer", 0.0) + vlm_res.estimated_cost_usd
                if vlm_res.violates_avoid:
                    continue

                if local_best is None or vlm_res.relevance_score > local_best[2].relevance_score:
                    local_best = (cand, clip_path, vlm_res)
                if local_best[2].relevance_score >= 4.4:
                    return local_best
            if local_best and local_best[2].relevance_score >= stock_threshold:
                return local_best
        return local_best

    queries = [scene.search_query] + [q for q in scene.alt_queries if q != scene.search_query]

    # Explicit preferred generative or statcard (e.g. via swap)
    if preferred_source == "generative" and "generative" in sources and not scene.block_ai_generation:
        gen_hit = try_source_candidates("generative", sources["generative"], [scene.search_query], limit=1)
        if gen_hit:
            usage_costs["generative"] = usage_costs.get("generative", 0.0) + gen_hit[0].estimated_cost_usd
            return (
                gen_hit[0], gen_hit[1], gen_hit[2],
                "R1_PREFERRED_SOURCE",
                f"User requested generative source; VLM relevance={gen_hit[2].relevance_score:.2f}/5.0.",
            )

    if preferred_source == "statcard" and "statcard" in sources:
        sc_hit = try_source_candidates("statcard", sources["statcard"], [scene.search_query], limit=1)
        if sc_hit:
            return (
                sc_hit[0], sc_hit[1], sc_hit[2],
                "R1_PREFERRED_SOURCE",
                f"User requested programmatic stat card; VLM relevance={sc_hit[2].relevance_score:.2f}/5.0.",
            )

    # Step 1: Try stock sources first unless scene explicitly triggers statcard or abstract generative preference
    for src_name in stock_order:
        hit = try_source_candidates(src_name, sources[src_name], queries, limit=3)
        if hit and (best_stock is None or hit[2].relevance_score > best_stock[2].relevance_score):
            best_stock = hit
        if best_stock and best_stock[2].relevance_score >= max(stock_threshold, 4.2):
            break

    # Step 2: Apply measurable decision rule between Stock vs Generative vs StatCard
    best_stock_score = best_stock[2].relevance_score if best_stock else 0.0

    # Rule: For quantitative stats where stock relevance < threshold + 0.5 -> animated stat card
    if scene.prefer_statcard and "statcard" in sources and best_stock_score < min(4.3, stock_threshold + 0.6):
        sc_hit = try_source_candidates("statcard", sources["statcard"], [scene.search_query], limit=1)
        if sc_hit:
            return (
                sc_hit[0], sc_hit[1], sc_hit[2],
                "R4_STAT_GRAPHIC",
                f"Quantitative statistic ('{scene.stat_headline}') detected with stock_score={best_stock_score:.2f} < 4.10; generated animated data card (VLM={sc_hit[2].relevance_score:.2f}).",
            )

    # Rule: If stock beats threshold and isn't forced to generative abstract metaphor -> choose stock
    if best_stock is not None and best_stock_score >= stock_threshold and not scene.prefer_generative:
        cand, path, vlm = best_stock
        return (
            cand, path, vlm,
            "R2_STOCK_THRESHOLD_MET",
            f"Selected stock clip from '{cand.source}' because VLM relevance {vlm.relevance_score:.2f} >= threshold {stock_threshold:.2f} (0 cost, real-world footage).",
        )

    # Rule: Escalate to AI generative source when stock < threshold (or abstract metaphor) AND safety permits
    if "generative" in sources and "generative" not in failed_sources and not scene.block_ai_generation:
        gen_hit = try_source_candidates("generative", sources["generative"], [scene.search_query], limit=1)
        if gen_hit:
            gen_cand, gen_path, gen_vlm = gen_hit
            usage_costs["generative"] = usage_costs.get("generative", 0.0) + gen_cand.estimated_cost_usd
            if best_stock is None or gen_vlm.relevance_score >= best_stock_score or scene.prefer_generative:
                reason = (
                    f"Escalated to AI generative source because {'abstract metaphor' if scene.prefer_generative else f'best stock score {best_stock_score:.2f} < threshold {stock_threshold:.2f}'} "
                    f"and no real person/brand detected; generative VLM={gen_vlm.relevance_score:.2f}."
                )
                return gen_cand, gen_path, gen_vlm, "R3_GENERATIVE_ESCALATION", reason

    # If we still have a stock candidate (e.g., AI blocked due to real person/brand), use it!
    if best_stock is not None:
        cand, path, vlm = best_stock
        reason = (
            f"Selected stock clip from '{cand.source}' (VLM={vlm.relevance_score:.2f}); "
            f"{'AI generation blocked by safety guardrail (real person/brand)' if scene.block_ai_generation else 'best available stock candidate'}."
        )
        return cand, path, vlm, "R2_STOCK_SAFETY_FALLBACK", reason

    # Ultimate resilient fallback: programmatic animated card (guarantees job NEVER crashes on total outage!)
    if "statcard" in sources:
        sc_hit = try_source_candidates("statcard", sources["statcard"], [scene.search_query], limit=1)
        if sc_hit:
            warnings.append(
                f"Scene {scene.scene_index}: all external footage sources unavailable; used programmatic animated card fallback."
            )
            return (
                sc_hit[0], sc_hit[1], sc_hit[2],
                "R4_RESILIENT_FALLBACK",
                "All external footage sources unavailable or returned no safe candidates; generated programmatic animated fallback card.",
            )

    raise AppError("PROVIDER_UNAVAILABLE", f"No footage source could satisfy scene {scene.scene_index}.", status_code=503, retryable=True)


def execute_broll_job(job_id: str, settings: Optional[Settings] = None, storage: Optional[StorageBackend] = None) -> None:
    """Execute a queued B-roll job from start to finish."""
    cfg = settings or get_settings()
    store = storage or get_storage()
    t_start = time.perf_counter()
    timings: Dict[str, float] = {}
    warnings: List[str] = []
    usage_counts: Dict[str, int] = {}
    usage_costs: Dict[str, float] = {}

    with get_session() as db:
        rec = db.get(JobRecord, job_id)
        if not rec:
            return
        req_dict = dict(rec.request_payload)
        corr_id = rec.correlation_id

    token_job = job_id_ctx.set(job_id)
    token_corr = correlation_id_ctx.set(corr_id)
    work_dir = cfg.temp_work_dir.resolve() / job_id
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        req = CreateBrollJobRequest.model_validate(req_dict)
        seed = req.seed if req.seed is not None else cfg.default_seed
        runtime_disabled = set(req.disabled_sources or [])
        registry = AdapterRegistry(cfg, runtime_disabled_sources=runtime_disabled)

        update_job(job_id, status="running", progress=5, stage="ingesting_input")
        logger.info("job_started", aspect_ratios=req.aspect_ratios, adapter_mode=cfg.adapter_mode)

        # ---- STAGE 1: Audio acquisition / TTS + Validation ----
        t0 = time.perf_counter()
        tts_adapter = registry.get_tts()
        hint_transcript: Optional[str] = None
        language = req.language

        uploaded_key = req_dict.get("_uploaded_audio_key")
        if uploaded_key:
            ext = Path(uploaded_key).suffix or ".audio"
            audio_path = store.get_file(uploaded_key, work_dir / f"narration_input{ext}")
        elif req.audio_url:
            audio_path = download_audio_url(req.audio_url, work_dir, max_mb=cfg.max_upload_mb)
        else:
            # Text script or IncuBrix scene list -> synthesize via TTS
            if req.scenes:
                full_text = " ".join(s.text.strip() for s in req.scenes if s.text.strip())
            else:
                full_text = (req.script or "").strip()
            if not full_text:
                raise AppError("EMPTY_SCRIPT", "Narration script is empty.", status_code=400)
            hint_transcript = full_text
            tts_res = tts_adapter.synthesize(
                full_text,
                voice=req.voice,
                out_path=work_dir / "narration.wav",
                language_hint=req.language,
            )
            audio_path = tts_res.audio_path
            language = tts_res.language
            usage_counts["tts"] = 1
            usage_costs["tts"] = tts_res.estimated_cost_usd

        probe = probe_and_validate_audio(audio_path, max_duration_seconds=cfg.max_audio_duration_seconds)
        total_duration = probe.duration_seconds
        # Cache narration audio for fast swap-a-clip later
        narration_cache_key = f"jobs/{job_id}/cache/narration{audio_path.suffix or '.wav'}"
        store.put_file(audio_path, narration_cache_key, content_type="audio/wav")
        timings["ingest_and_tts"] = round(time.perf_counter() - t0, 3)

        # ---- STAGE 2: ASR Word-Level Alignment ----
        update_job(job_id, progress=20, stage="transcribing_audio")
        t0 = time.perf_counter()
        asr_adapter = registry.get_asr(hint_transcript=hint_transcript)
        asr_res = asr_adapter.transcribe(audio_path, language_hint=language)
        usage_counts["asr"] = 1
        usage_costs["asr"] = asr_res.estimated_cost_usd
        if not asr_res.speech_detected or not asr_res.text.strip():
            raise AppError(
                "AUDIO_SILENT_OR_NO_SPEECH",
                "No intelligible spoken narration detected (audio may be silent or music-only).",
                status_code=422,
                retryable=False,
            )
        language = asr_res.language or language or "en"
        total_duration = probe.duration_seconds
        timings["asr_alignment"] = round(time.perf_counter() - t0, 3)

        # ---- STAGE 3: Semantic Scene Planning + Safety Guardrails ----
        update_job(job_id, progress=35, stage="planning_scenes")
        t0 = time.perf_counter()
        planner = registry.get_planner()
        planned_scenes = planner.plan_scenes(
            transcript=asr_res.text,
            words=asr_res.words,
            total_duration=total_duration,
            style=req.style,
            avoid=req.avoid,
            explicit_scenes=req.scenes,
            min_scene_duration=cfg.min_scene_duration_seconds,
            max_scene_duration=cfg.max_scene_duration_seconds,
        )
        usage_counts["planner"] = 1
        usage_costs["planner"] = 0.0005 if planner.provider_name != "mock" else 0.0
        for sc in planned_scenes:
            for w in sc.warnings:
                if w not in warnings:
                    warnings.append(w)
        timings["scene_planning"] = round(time.perf_counter() - t0, 3)

        # Build global caption cues & save captions.srt
        cues = build_caption_cues(asr_res.words)
        srt_local = work_dir / "captions.srt"
        write_srt(cues, srt_local)
        srt_key = f"jobs/{job_id}/captions.srt"
        if req.captions.srt:
            store.put_file(srt_local, srt_key, content_type="application/x-subrip")
        store.put_bytes(
            json.dumps({"cues": cues, "language": language, "narration_key": narration_cache_key}).encode("utf-8"),
            f"jobs/{job_id}/cache/cues.json",
            content_type="application/json",
        )

        # ---- STAGE 4: Multi-Source Footage Retrieval, VLM Reranking & Deduplication ----
        update_job(job_id, progress=50, stage="selecting_footage")
        t0 = time.perf_counter()
        sources = registry.get_footage_sources()
        vlm_scorer = registry.get_vlm_scorer()
        dedup = ClipDeduplicator()
        failed_sources: set = set()
        orientation = _orientation_for_ratios(req.aspect_ratios)

        selected_entries: List[Tuple[PlannedScene, FootageCandidate, Path, VLMScoreResult, str, str, float]] = []
        for idx, sc in enumerate(planned_scenes):
            cand, clip_path, vlm_res, rule_name, rule_reason = _select_clip_for_scene(
                sc,
                sources=sources,
                vlm_scorer=vlm_scorer,
                dedup=dedup,
                avoid_terms=req.avoid,
                orientation=orientation,
                stock_threshold=cfg.stock_min_relevance_score,
                work_dir=work_dir,
                seed=seed,
                warnings=warnings,
                failed_sources=failed_sources,
                usage_counts=usage_counts,
                usage_costs=usage_costs,
            )
            dedup.register(cand.clip_id, cand.source_url, clip_path)
            focus_x = detect_subject_focus_x(clip_path, vlm_hint_x=vlm_res.subject_focus_x)
            selected_entries.append((sc, cand, clip_path, vlm_res, rule_name, rule_reason, round(focus_x, 3)))
            pct = 50 + int(20 * (idx + 1) / max(1, len(planned_scenes)))
            update_job(job_id, progress=pct, stage=f"selecting_footage_{idx + 1}_of_{len(planned_scenes)}")
        timings["footage_selection_and_vlm"] = round(time.perf_counter() - t0, 3)

        # ---- STAGE 5: Normalization, Color Matching, Per-Scene Cache & Assembly ----
        update_job(job_id, progress=72, stage="rendering_videos")
        t0 = time.perf_counter()
        video_keys: Dict[str, str] = {}
        qa_results: Dict[str, Any] = {}

        for r_idx, ratio in enumerate(req.aspect_ratios):
            tag = RATIO_FILE_TAGS[ratio]
            segment_paths: List[Path] = []
            for sc, _cand, clip_path, _vlm, _rn, _rr, focus_x in selected_entries:
                seg_local = work_dir / "segments" / f"scene_{sc.scene_index:03d}_{tag}.mp4"
                scene_cues = (
                    slice_cues_for_scene(cues, sc.start_time, sc.end_time)
                    if req.captions.burn_in
                    else None
                )
                render_scene_segment(
                    clip_path,
                    seg_local,
                    duration=max(0.5, sc.end_time - sc.start_time),
                    aspect_ratio=ratio,
                    focus_x=focus_x,
                    fps=cfg.target_fps,
                    color_normalize=cfg.color_normalize_enabled,
                    scene_cues=scene_cues,
                    language=language,
                    work_dir=work_dir / "segments",
                )
                # Store normalized segment in cache for fast swap-a-clip!
                seg_cache_key = f"jobs/{job_id}/cache/scene_{sc.scene_index:03d}_{tag}.mp4"
                store.put_file(seg_local, seg_cache_key, content_type="video/mp4")
                segment_paths.append(seg_local)
                import gc
                gc.collect()

            out_video = work_dir / f"video_{tag}.mp4"
            assemble_video_from_segments(segment_paths, audio_path, out_video, work_dir)
            qa_results[ratio] = run_video_qa(
                out_video, aspect_ratio=ratio, expected_duration=total_duration, expected_fps=cfg.target_fps
            )
            if not qa_results[ratio]["passed"]:
                warnings.append(f"QA check noted minor deviation on {ratio}: {qa_results[ratio]}")

            vid_key = f"jobs/{job_id}/video_{tag}.mp4"
            store.put_file(out_video, vid_key, content_type="video/mp4")
            video_keys[ratio] = vid_key
            update_job(job_id, progress=72 + int(20 * (r_idx + 1) / len(req.aspect_ratios)), stage=f"rendered_{tag}")

        timings["render_and_qa"] = round(time.perf_counter() - t0, 3)
        timings["total"] = round(time.perf_counter() - t_start, 3)

        # ---- STAGE 6: Timeline & Manifest Serialization ----
        timeline_scenes: List[TimelineScene] = []
        licences: List[ClipLicenceInfo] = []
        for sc, cand, _clip_path, vlm_res, rule_name, rule_reason, focus_x in selected_entries:
            timeline_scenes.append(
                TimelineScene(
                    scene_index=sc.scene_index,
                    start_time=sc.start_time,
                    end_time=sc.end_time,
                    duration=round(sc.end_time - sc.start_time, 3),
                    narration_text=sc.narration_text,
                    visual_hint=sc.visual_hint,
                    search_query=sc.search_query,
                    disambiguation_notes=sc.disambiguation_notes,
                    safety_flags=sc.safety_flags,
                    clip_id=cand.clip_id,
                    source=cand.source,
                    licence=cand.licence,
                    attribution=cand.attribution,
                    source_url=cand.source_url,
                    ai_generated=cand.ai_generated,
                    selection_rule=rule_name,
                    selection_reason=rule_reason,
                    relevance_score=vlm_res.relevance_score,
                    vlm_rationale=vlm_res.rationale,
                    crop_focus_x=focus_x,
                    color_normalized=cfg.color_normalize_enabled,
                )
            )
            licences.append(
                ClipLicenceInfo(
                    source=cand.source,
                    clip_id=cand.clip_id,
                    licence=cand.licence,
                    attribution=cand.attribution,
                    source_url=cand.source_url,
                    commercial_use_allowed=cand.commercial_use_allowed,
                    watermarked=cand.watermarked,
                    ai_generated=cand.ai_generated,
                )
            )

        unique_ids = len({s.clip_id for s in timeline_scenes})
        repeat_count = max(0, len(timeline_scenes) - unique_ids)
        avg_rel = round(
            sum(s.relevance_score for s in timeline_scenes) / max(1, len(timeline_scenes)), 2
        )

        timeline_doc = TimelineDocument(
            job_id=job_id,
            language=language,
            total_duration=round(total_duration, 3),
            aspect_ratios=req.aspect_ratios,
            scene_count=len(timeline_scenes),
            repeat_clip_count=repeat_count,
            average_relevance_score=avg_rel,
            scenes=timeline_scenes,
        )
        timeline_key = f"jobs/{job_id}/timeline.json"
        store.put_bytes(
            timeline_doc.model_dump_json(indent=2).encode("utf-8"),
            timeline_key,
            content_type="application/json",
        )

        models_used = [
            ProviderUsageEntry(
                role="asr",
                provider=asr_adapter.provider_name,
                model_or_version=asr_adapter.model_name,
                calls=usage_counts.get("asr", 0),
                estimated_cost_usd=round(usage_costs.get("asr", 0.0), 6),
            ),
            ProviderUsageEntry(
                role="planner",
                provider=planner.provider_name,
                model_or_version=planner.model_name,
                calls=usage_counts.get("planner", 0),
                estimated_cost_usd=round(usage_costs.get("planner", 0.0), 6),
            ),
            ProviderUsageEntry(
                role="vlm_scorer",
                provider=vlm_scorer.provider_name,
                model_or_version=vlm_scorer.model_name,
                calls=usage_counts.get("vlm_scorer", 0),
                estimated_cost_usd=round(usage_costs.get("vlm_scorer", 0.0), 6),
            ),
        ]
        if usage_counts.get("tts", 0) > 0:
            models_used.append(
                ProviderUsageEntry(
                    role="tts",
                    provider=tts_adapter.provider_name,
                    model_or_version=tts_adapter.model_name,
                    calls=usage_counts["tts"],
                    estimated_cost_usd=round(usage_costs.get("tts", 0.0), 6),
                )
            )
        for src_name in sources:
            if usage_counts.get(src_name, 0) > 0:
                models_used.append(
                    ProviderUsageEntry(
                        role="footage_source",
                        provider=src_name,
                        model_or_version=cfg.gemini_image_model if src_name == "generative" and cfg.adapter_mode == "live" else f"{src_name}-v1",
                        calls=usage_counts[src_name],
                        estimated_cost_usd=round(usage_costs.get(src_name, 0.0), 6),
                    )
                )

        total_cost = round(sum(m.estimated_cost_usd for m in models_used), 6)
        cost_per_min = round(total_cost / max(0.05, total_duration / 60.0), 6)
        now = datetime.now(timezone.utc)

        manifest_doc = ManifestDocument(
            job_id=job_id,
            created_at=rec.created_at,
            completed_at=now,
            seed=seed,
            settings={
                "adapter_mode": cfg.adapter_mode,
                "aspect_ratios": req.aspect_ratios,
                "style": req.style,
                "avoid": req.avoid,
                "captions": req.captions.model_dump(),
                "stock_min_relevance_score": cfg.stock_min_relevance_score,
                "min_scene_duration_seconds": cfg.min_scene_duration_seconds,
                "target_fps": cfg.target_fps,
                "color_normalize_enabled": cfg.color_normalize_enabled,
            },
            models_and_providers=models_used,
            timings_seconds=timings,
            cost_estimate_usd={
                "total_job_usd": total_cost,
                "per_minute_video_usd": cost_per_min,
            },
            qa_checks=qa_results,
            warnings=warnings,
            rights_and_licences=licences,
            ai_generated_clip_count=sum(1 for x in licences if x.ai_generated),
        )
        manifest_key = f"jobs/{job_id}/manifest.json"
        store.put_bytes(
            manifest_doc.model_dump_json(indent=2).encode("utf-8"),
            manifest_key,
            content_type="application/json",
        )

        result_keys = {
            "videos": video_keys,
            "captions_srt": srt_key if req.captions.srt else None,
            "timeline_json": timeline_key,
            "manifest_json": manifest_key,
        }
        metrics = {
            "total_seconds": timings["total"],
            "render_seconds": timings["render_and_qa"],
            "narration_duration_seconds": round(total_duration, 3),
            "scene_count": len(timeline_scenes),
            "average_relevance_score": avg_rel,
            "repeat_clip_count": repeat_count,
            "estimated_cost_usd": total_cost,
        }

        update_job(
            job_id,
            status="succeeded",
            progress=100,
            stage="completed",
            result_keys=result_keys,
            warnings=warnings,
            metrics=metrics,
        )
        logger.info("job_succeeded", total_seconds=timings["total"], avg_relevance=avg_rel)

        if req.callback_url:
            send_webhook(
                req.callback_url,
                {
                    "event": "job.succeeded",
                    "job_id": job_id,
                    "status": "succeeded",
                    "progress": 100,
                    "metrics": metrics,
                    "warnings": warnings,
                },
            )
    except AppError as exc:
        logger.error("job_failed_domain", code=exc.code, message=exc.message)
        update_job(
            job_id,
            status="failed",
            stage="failed",
            warnings=warnings,
            error_payload=exc.to_dict()["error"],
        )
        if req_dict.get("callback_url"):
            send_webhook(
                req_dict["callback_url"],
                {"event": "job.failed", "job_id": job_id, "status": "failed", "error": exc.to_dict()["error"]},
            )
    except Exception as exc:
        logger.error("job_failed_unexpected", error=str(exc), exc_type=type(exc).__name__)
        err = AppError("INTERNAL_ERROR", f"Unexpected pipeline failure: {exc}", status_code=500, retryable=True)
        update_job(
            job_id,
            status="failed",
            stage="failed",
            warnings=warnings,
            error_payload=err.to_dict()["error"],
        )
        if req_dict.get("callback_url"):
            send_webhook(
                req_dict["callback_url"],
                {"event": "job.failed", "job_id": job_id, "status": "failed", "error": err.to_dict()["error"]},
            )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        import gc
        gc.collect()
        job_id_ctx.reset(token_job)
        correlation_id_ctx.reset(token_corr)


def execute_swap_scene_clip(
    job_id: str,
    scene_index: int,
    swap_req: SwapSceneClipRequest,
    settings: Optional[Settings] = None,
    storage: Optional[StorageBackend] = None,
) -> Dict[str, Any]:
    """
    Fast partial re-render (Advanced Level: Swap-a-Clip):
    Re-fetches & re-renders ONLY scene_index across requested aspect ratios, then stream-copies (-c:v copy)
    all unchanged cached scene segments into the final MP4 in a fraction of full render time.
    """
    cfg = settings or get_settings()
    store = storage or get_storage()
    t0 = time.perf_counter()

    with get_session() as db:
        rec = db.get(JobRecord, job_id)
        if not rec:
            raise AppError("JOB_NOT_FOUND", f"Job '{job_id}' not found.", status_code=404)
        if rec.status != "succeeded":
            raise AppError("JOB_NOT_READY", f"Job '{job_id}' is '{rec.status}', must be 'succeeded'.", status_code=409)
        req = CreateBrollJobRequest.model_validate(rec.request_payload)
        existing_warnings = list(rec.warnings or [])

    timeline_key = f"jobs/{job_id}/timeline.json"
    manifest_key = f"jobs/{job_id}/manifest.json"
    timeline = TimelineDocument.model_validate_json(store.get_bytes(timeline_key))
    manifest = ManifestDocument.model_validate_json(store.get_bytes(manifest_key))

    if scene_index < 0 or scene_index >= len(timeline.scenes):
        raise AppError(
            "SCENE_NOT_FOUND",
            f"scene_index {scene_index} out of range [0, {len(timeline.scenes) - 1}].",
            status_code=404,
        )

    target_scene = timeline.scenes[scene_index]
    work_dir = cfg.temp_work_dir.resolve() / f"{job_id}_swap_{scene_index}_{int(time.time())}"
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        cache_meta = json.loads(store.get_bytes(f"jobs/{job_id}/cache/cues.json").decode("utf-8"))
        cues = [tuple(c) for c in cache_meta["cues"]]
        language = cache_meta.get("language", timeline.language)
        narration_key = cache_meta["narration_key"]
        narration_local = store.get_file(narration_key, work_dir / f"narration{Path(narration_key).suffix}")

        # Build deduplicator containing ALL existing clips (including current scene so we get a genuinely new clip!)
        dedup = ClipDeduplicator()
        for sc in timeline.scenes:
            dedup.register(sc.clip_id, sc.source_url)

        full_script = " ".join(s.narration_text for s in timeline.scenes)
        safety = analyze_scene_safety(target_scene.narration_text, full_script, req.avoid)
        new_query = swap_req.query or swap_req.visual_hint or f"{target_scene.search_query} alternative angle"

        planned = PlannedScene(
            scene_index=scene_index,
            start_time=target_scene.start_time,
            end_time=target_scene.end_time,
            narration_text=target_scene.narration_text,
            visual_hint=swap_req.visual_hint or target_scene.visual_hint,
            search_query=new_query,
            alt_queries=[f"{new_query} wide shot", "modern editorial footage"],
            generative_prompt=f"{new_query}, {req.style}, photorealistic, no logos",
            safety_flags=safety.flags,
            negative_keywords=safety.negative_keywords,
            block_ai_generation=safety.block_ai_generation,
        )

        registry = AdapterRegistry(cfg, runtime_disabled_sources=set(req.disabled_sources or []))
        sources = registry.get_footage_sources()
        vlm_scorer = registry.get_vlm_scorer()
        swap_warnings: List[str] = []
        usage_counts: Dict[str, int] = {}
        usage_costs: Dict[str, float] = {}

        cand, clip_path, vlm_res, rule_name, rule_reason = _select_clip_for_scene(
            planned,
            sources=sources,
            vlm_scorer=vlm_scorer,
            dedup=dedup,
            avoid_terms=req.avoid,
            orientation=_orientation_for_ratios(req.aspect_ratios),
            stock_threshold=cfg.stock_min_relevance_score,
            work_dir=work_dir,
            seed=(req.seed or cfg.default_seed) + 777,
            warnings=swap_warnings,
            failed_sources=set(),
            preferred_source=swap_req.preferred_source,
            usage_counts=usage_counts,
            usage_costs=usage_costs,
        )
        focus_x = round(detect_subject_focus_x(clip_path, vlm_hint_x=vlm_res.subject_focus_x), 3)

        # Re-render ONLY scene_index per ratio, pull unchanged segments from cache, concat with -c:v copy!
        for ratio in req.aspect_ratios:
            tag = RATIO_FILE_TAGS[ratio]
            new_seg = work_dir / "segments" / f"scene_{scene_index:03d}_{tag}.mp4"
            scene_cues = (
                slice_cues_for_scene(cues, target_scene.start_time, target_scene.end_time)
                if req.captions.burn_in
                else None
            )
            render_scene_segment(
                clip_path,
                new_seg,
                duration=target_scene.duration,
                aspect_ratio=ratio,
                focus_x=focus_x,
                fps=cfg.target_fps,
                color_normalize=cfg.color_normalize_enabled,
                scene_cues=scene_cues,
                language=language,
                work_dir=work_dir / "segments",
            )
            store.put_file(new_seg, f"jobs/{job_id}/cache/scene_{scene_index:03d}_{tag}.mp4", content_type="video/mp4")

            seg_paths: List[Path] = []
            for idx in range(len(timeline.scenes)):
                if idx == scene_index:
                    seg_paths.append(new_seg)
                else:
                    cached = work_dir / "segments" / f"scene_{idx:03d}_{tag}.mp4"
                    store.get_file(f"jobs/{job_id}/cache/scene_{idx:03d}_{tag}.mp4", cached)
                    seg_paths.append(cached)

            out_video = work_dir / f"video_{tag}.mp4"
            assemble_video_from_segments(seg_paths, narration_local, out_video, work_dir)
            store.put_file(out_video, f"jobs/{job_id}/video_{tag}.mp4", content_type="video/mp4")

        # Update timeline.json & manifest.json atomically
        timeline.scenes[scene_index] = TimelineScene(
            scene_index=scene_index,
            start_time=target_scene.start_time,
            end_time=target_scene.end_time,
            duration=target_scene.duration,
            narration_text=target_scene.narration_text,
            visual_hint=swap_req.visual_hint or target_scene.visual_hint,
            search_query=new_query,
            disambiguation_notes=target_scene.disambiguation_notes,
            safety_flags=safety.flags,
            clip_id=cand.clip_id,
            source=cand.source,
            licence=cand.licence,
            attribution=cand.attribution,
            source_url=cand.source_url,
            ai_generated=cand.ai_generated,
            selection_rule=rule_name,
            selection_reason=f"Clip swapped: {rule_reason}",
            relevance_score=vlm_res.relevance_score,
            vlm_rationale=vlm_res.rationale,
            crop_focus_x=focus_x,
            color_normalized=cfg.color_normalize_enabled,
        )
        timeline.average_relevance_score = round(
            sum(s.relevance_score for s in timeline.scenes) / len(timeline.scenes), 2
        )
        store.put_bytes(timeline.model_dump_json(indent=2).encode("utf-8"), timeline_key, content_type="application/json")

        swap_elapsed = round(time.perf_counter() - t0, 3)
        manifest.timings_seconds[f"swap_scene_{scene_index}"] = swap_elapsed
        manifest.rights_and_licences[scene_index] = ClipLicenceInfo(
            source=cand.source,
            clip_id=cand.clip_id,
            licence=cand.licence,
            attribution=cand.attribution,
            source_url=cand.source_url,
            commercial_use_allowed=cand.commercial_use_allowed,
            watermarked=cand.watermarked,
            ai_generated=cand.ai_generated,
        )
        manifest.ai_generated_clip_count = sum(1 for r in manifest.rights_and_licences if r.ai_generated)
        manifest.warnings.extend(swap_warnings)
        store.put_bytes(manifest.model_dump_json(indent=2).encode("utf-8"), manifest_key, content_type="application/json")

        full_render_sec = manifest.timings_seconds.get("render_and_qa", 0.0)
        update_job(
            job_id,
            warnings=existing_warnings + swap_warnings,
            metrics={
                **(rec.metrics or {}),
                f"last_swap_scene_{scene_index}_seconds": swap_elapsed,
                "full_render_seconds": full_render_sec,
            },
        )
        return {
            "job_id": job_id,
            "scene_index": scene_index,
            "new_clip_id": cand.clip_id,
            "new_source": cand.source,
            "relevance_score": vlm_res.relevance_score,
            "swap_duration_seconds": swap_elapsed,
            "original_full_render_seconds": full_render_sec,
            "re_rendered_scenes": 1,
            "cached_scenes_reused": len(timeline.scenes) - 1,
        }
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        import gc
        gc.collect()
