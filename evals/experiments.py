"""Run the experiment plan (vault Testing.md → "Experiments") until done or rate-limited.

    uv run python evals/experiments.py                       # ipl + sales, default model
    uv run python evals/experiments.py --datasets ipl --only baseline,e1-attempts-1
    uv run python evals/experiments.py --model groq:openai/gpt-oss-120b --only baseline --tag e5

Idempotent: a finished run (evals/results/<dataset>_<label>_*.json) is skipped, a partial one is
resumed, so on a free tier with a daily token limit you just run it again the next day. It stops
at the first rate limit, since every later run would hit the same limit.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals import run_evals

RATE_LIMITED = 3

# Headline numbers first: baseline and self-correction on both datasets, then the ablations
PLAN: list[tuple[str, list[str]]] = [
    ("baseline", []),
    ("e1-attempts-1", ["--attempts", "1"]),
    ("e2-no-comments", ["--no-comments"]),
    ("e3-fewshots-0", ["--fewshots", "0"]),
    ("e4-retrieval", ["--retrieval", "--embedding-model", "hashing"]),
]


def latest(dataset: str, label: str) -> dict | None:
    files = sorted(run_evals.RESULTS_DIR.glob(f"{dataset}_{label}_*.json"))
    if not files:
        return None
    data = json.loads(files[-1].read_text())
    return {"path": files[-1], "partial": data["meta"].get("partial", False)}


async def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--datasets", default="ipl,sales")
    p.add_argument("--only", help="comma-separated labels from the plan")
    p.add_argument("--model", help="LLM_MODEL override for every run")
    p.add_argument("--tag", help="label prefix, e.g. e5 → e5-baseline")
    args = p.parse_args(argv)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    plan = [s for s in PLAN if not args.only or s[0] in args.only.split(",")]
    for label, extra in plan:
        for dataset in datasets:
            full = f"{args.tag}-{label}" if args.tag else label
            found = latest(dataset, full)
            if found and not found["partial"]:
                print(f"✓ {dataset} {full}: done ({found['path'].name})")
                continue
            run_argv = ["--dataset", dataset, "--label", full, *extra]
            run_argv += ["--model", args.model] if args.model else []
            run_argv += ["--resume", str(found["path"])] if found else []
            print(f"\n▶ {dataset} {full}")
            code = await run_evals.main(run_argv)
            if code == RATE_LIMITED:
                print("\n⏸ rate limited: run this command again later to continue")
                return RATE_LIMITED
            if code:
                return code
    print("\nall planned runs are done")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
