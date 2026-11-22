"""Compute agreement between independent VLM scores and human 1-5 ratings (Outstanding Level).
Pure Python implementation with zero DLL / external library dependency for rock-solid portability.
"""

import argparse
import json
import math
from pathlib import Path
import sys
from typing import List, Tuple


def _rank(values: List[float]) -> List[float]:
    """Compute average ranks for tied values."""
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    n = len(values)
    while i < n:
        j = i
        while j < n - 1 and indexed[j + 1][1] == indexed[j][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0  # 1-based ranking
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def _pearson(x: List[float], y: List[float]) -> float:
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den_x = sum((xi - mx) ** 2 for xi in x)
    den_y = sum((yi - my) ** 2 for yi in y)
    den = math.sqrt(den_x * den_y)
    return num / den if den > 1e-9 else 0.0


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
    n = len(rows)

    diffs = [abs(v - h) for v, h in zip(vlm, human)]
    mae = sum(diffs) / n
    rmse = math.sqrt(sum(d ** 2 for d in diffs) / n)

    pr = _pearson(vlm, human)
    sr = _pearson(_rank(vlm), _rank(human))

    within_0_5 = sum(1 for d in diffs if d <= 0.5) / n * 100.0
    within_1_0 = sum(1 for d in diffs if d <= 1.0) / n * 100.0
    h_4_5 = sum(1 for h in human if h >= 4.0) / n * 100.0
    h_3_plus = sum(1 for h in human if h >= 3.0) / n * 100.0

    summary = {
        "total_scenes_evaluated": n,
        "mean_human_score": round(sum(human) / n, 3),
        "mean_vlm_score": round(sum(vlm) / n, 3),
        "pct_human_4_or_5": round(h_4_5, 2),
        "pct_human_3_or_above_coverage": round(h_3_plus, 2),
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "within_0_5_agreement_pct": round(within_0_5, 2),
        "within_1_0_agreement_pct": round(within_1_0, 2),
        "pearson_r": round(pr, 4),
        "spearman_rho": round(sr, 4),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
