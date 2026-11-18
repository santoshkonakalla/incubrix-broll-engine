"""Export exact OpenAPI 3.1 contract from FastAPI to openapi.yaml (Integration Standard #2)."""

from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.main import app


def main() -> None:
    schema = app.openapi()
    schema["openapi"] = "3.1.0"
    out_path = ROOT / "openapi.yaml"
    out_path.write_text(yaml.safe_dump(schema, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"Wrote OpenAPI {schema['openapi']} spec with {len(schema['paths'])} paths -> {out_path}")


if __name__ == "__main__":
    main()
