"""Compute agreement between independent VLM scores and human 1-5 ratings (Outstanding Level)."""

import argparse
import json
from pathlib import Path
import sys
from typing import List

import numpy as np
from scipy.stats import pearsonr, spearmanr


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VLM vs Human Relevance Score Agreement")
    parser.add_argument("--ratings-json", required=True, type=Path, help="JSON list of {video, scene_index, vlm_score, human_score}")
    parser.add_argument("--out", required=True, type=Path, help="Output summary JSON path")
    args = parser.parse_args()

    rows = json.loads(args.ratings_json.read_text(encoding="utf-8"))
    if len(rows) < 3:
        print("Need at least 3 rated scenes to compute correlation.")
        sys.exit(1)

    vlm: List[float] = [float(r["vlm_score"]) for r in rows]
    human: List[float] = [float(r["human_score"]) for r in rows]

    v_arr = np.array(vlm, dtype=float)
    h_arr = np.array(human, dtype=float)
    diffs = np.abs(v_arr - h_arr)

    pr, p_val = pearsonr(v_arr, h_arr) if np.std(v_arr) > 0 and np.std(h_arr) > 0 else (0.0, 1.0)
    sr, s_val = spearmanr(v_arr, h_arr) if np.std(v_arr) > 0 and np.std(h_arr) > 0 else (0.0, 1.0)

    summary = {
        "total_scenes_evaluated": len(rows),
        "mean_human_score": round(float(np.mean(h_arr)), 3),
        "mean_vlm_score": round(float(np.mean(v_arr)), 3),
        "pct_human_4_or_5": round(float(np.mean(h_arr >= 4.0) * 100.0), 2),
        "pct_human_3_or_above_coverage": round(float(np.mean(h_arr >= 3.0) * 100.0), 2),
        "mae": round(float(np.mean(diffs)), 3),
        "rmse": round(float(np.sqrt(np.mean(diffs ** 2))), 3),
        "within_0_5_agreement_pct": round(float(np.mean(diffs <= 0.5) * 100.0), 2),
        "within_1_0_agreement_pct": round(float(np.mean(diffs <= 1.0) * 100.0), 2),
        "pearson_r": round(float(pr), 4),
        "pearson_p_value": round(float(p_val), 6),
        "spearman_rho": round(float(sr), 4),
        "spearman_p_value": round(float(s_val), 6),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
