import json
from pathlib import Path

for path in sorted(Path("test_pack/outputs").glob("*/timeline.json")):
    t = json.loads(path.read_text(encoding="utf-8"))
    scenes = t.get("scenes", [])
    print(f"=== {path.parent.name} ({len(scenes)} scenes) ===")
    for sc in scenes:
        q = sc.get("search_query", "")
        vlm = sc.get("relevance_score")
        src = sc.get("source")
        txt = sc.get("narration_text", "")[:45]
        print(f"  Scene {sc['scene_index']}: vlm={vlm} src={src} query='{q}' | '{txt}...'")
