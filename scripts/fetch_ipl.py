"""Download Cricsheet's IPL ball-by-ball JSON into datasets/ipl/raw/.

    uv run python scripts/fetch_ipl.py            # skip if raw/ already has match files
    uv run python scripts/fetch_ipl.py --force    # re-download and replace

Cricsheet data is ODC-BY licensed (https://cricsheet.org/register/). raw/ is gitignored.
"""

import argparse
import io
import shutil
import sys
import time
import zipfile
from pathlib import Path

import httpx

URL = "https://cricsheet.org/downloads/ipl_json.zip"
RAW_DIR = Path(__file__).resolve().parent.parent / "datasets" / "ipl" / "raw"
HEADERS = {"User-Agent": "query-smith/0.1 (+https://github.com/Vanshpanchal/query_smith)"}


def download(url: str, attempts: int = 3) -> bytes | None:
    """The zip's bytes, or None. Retries with backoff: Cricsheet sometimes answers CI runners
    with an error or a 200 HTML page instead of the zip."""
    for attempt in range(1, attempts + 1):
        print(f"downloading {url} … (attempt {attempt}/{attempts})")
        try:
            resp = httpx.get(url, follow_redirects=True, timeout=120, headers=HEADERS)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            problem = str(e)
        else:
            if zipfile.is_zipfile(io.BytesIO(resp.content)):
                return resp.content
            problem = f"not a zip (content-type: {resp.headers.get('content-type', 'unknown')})"
        print(f"✗ {problem}", file=sys.stderr)
        if attempt < attempts:
            time.sleep(5 * attempt)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    parser.add_argument("--url", default=URL)
    parser.add_argument("--dest", type=Path, default=RAW_DIR)
    args = parser.parse_args()

    existing = list(args.dest.glob("*.json")) if args.dest.is_dir() else []
    if existing and not args.force:
        print(f"✓ {len(existing)} match files already in {args.dest} (use --force to refresh)")
        return 0

    content = download(args.url)
    if content is None:
        return 1

    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        members = [m for m in zf.namelist() if m.endswith(".json") and "/" not in m]
        if not members:
            print("✗ zip contains no match .json files", file=sys.stderr)
            return 1
        if args.dest.exists():
            shutil.rmtree(args.dest)
        args.dest.mkdir(parents=True)
        # Only the flat *.json match files; README.txt etc. would change the dataset version
        for m in members:
            (args.dest / m).write_bytes(zf.read(m))

    print(f"✓ {len(members)} match files → {args.dest} ({len(content) / 1e6:.1f} MB zip)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
