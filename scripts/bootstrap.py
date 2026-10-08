"""Container start-up: migrate, then load every dataset in AUTOLOAD_DATASETS that isn't loaded.

    AUTOLOAD_DATASETS=ipl,sales python scripts/bootstrap.py

Idempotent, so it runs on every start (`scripts/start.sh`). For a dataset whose raw/ folder is
empty it first runs `scripts/fetch_<name>.py` if one exists (raw data is not in the image).
Schema cards are indexed with EMBEDDING_MODEL, or the key-free `hashing` model without a key.
"""

import os
import subprocess
import sys
from pathlib import Path

import psycopg

from ask_data.config import settings

ROOT = Path(__file__).resolve().parent.parent


def run(*args: str) -> None:
    print("→", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def loaded() -> set[str]:
    with psycopg.connect(settings.app_database_url) as conn:
        return {r[0] for r in conn.execute("SELECT name FROM app.datasets").fetchall()}


def main() -> int:
    run("scripts/migrate.py")
    wanted = [n.strip() for n in os.environ.get("AUTOLOAD_DATASETS", "").split(",") if n.strip()]
    have = loaded()
    for name in wanted:
        if name in have:
            print(f"✓ {name} already loaded", flush=True)
            continue
        raw = ROOT / settings.datasets_dir / name / "raw"
        fetch = ROOT / "scripts" / f"fetch_{name}.py"
        if not any(p.name != ".gitkeep" for p in raw.glob("*")) and fetch.exists():
            run(str(fetch.relative_to(ROOT)))
        run("scripts/load_dataset.py", name)
        model = [] if settings.google_api_key.strip() else ["--model", "hashing"]
        run("scripts/index_schema.py", name, *model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
