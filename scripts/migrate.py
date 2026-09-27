"""Apply db/migrations/*.sql in filename order, tracking what's already run in app._migrations."""
import os
import sys
from pathlib import Path

import psycopg

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "db" / "migrations"


def main() -> None:
    dsn = os.environ["APP_DATABASE_URL"]
    files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    if not files:
        print("No migrations found.")
        return

    with psycopg.connect(dsn, autocommit=False) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS app._migrations "
            "(filename text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
        )
        conn.commit()

        applied = {r[0] for r in conn.execute("SELECT filename FROM app._migrations").fetchall()}
        for path in files:
            if path.name in applied:
                print(f"skip  {path.name} (already applied)")
                continue
            print(f"apply {path.name}")
            sql = path.read_text()
            with conn.transaction():
                conn.execute(sql)
                conn.execute("INSERT INTO app._migrations (filename) VALUES (%s)", (path.name,))
        conn.commit()


if __name__ == "__main__":
    sys.exit(main())
