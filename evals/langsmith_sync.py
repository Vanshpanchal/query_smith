"""Optional LangSmith side of the evals (`run_evals.py --langsmith`, needs LANGSMITH_API_KEY).

1. The golden set becomes a LangSmith dataset `ask-data-<dataset>-golden` (created once; new
   questions are added, existing ones are left alone).
2. The results already scored locally are uploaded as an experiment named after the run's label.
   The agent is NOT run a second time: the target just returns each question's local record, and
   the evaluators read the scores from it. Local JSON stays the source of truth.
"""

import asyncio

from langsmith import Client, aevaluate


def dataset_name(dataset: str) -> str:
    return f"ask-data-{dataset}-golden"


def to_example(q: dict) -> dict:
    return {
        "inputs": {"question": q["question"]},
        "outputs": {
            "gold_sql": q.get("gold_sql"),
            "expected_outcome": q["expected_outcome"],
            "expected": q.get("expected"),
        },
        "metadata": {"id": q["id"], "category": q["category"], "split": q.get("split", "test")},
    }


def upload_golden(client: Client, dataset: str, golden: list[dict]) -> str:
    name = dataset_name(dataset)
    if client.has_dataset(dataset_name=name):
        ls_dataset = client.read_dataset(dataset_name=name)
        known = {e.inputs["question"] for e in client.list_examples(dataset_id=ls_dataset.id)}
    else:
        ls_dataset = client.create_dataset(name, description=f"AskData golden set: {dataset}")
        known = set()
    new = [to_example(q) for q in golden if q["question"] not in known]
    if new:
        client.create_examples(dataset_id=ls_dataset.id, examples=new)
    return name


def _score(key: str):
    def evaluator(run, example) -> dict:
        return {"key": key, "score": int(bool((run.outputs or {}).get(key)))}

    evaluator.__name__ = key
    return evaluator


async def upload_experiment(dataset: str, golden: list[dict], results: list[dict], meta: dict):
    client = Client()
    name = await asyncio.to_thread(upload_golden, client, dataset, golden)
    by_question = {r["question"]: r for r in results}

    async def replay(inputs: dict) -> dict:
        return by_question[inputs["question"]]

    selected = {q["question"] for q in golden}
    examples = [
        e for e in client.list_examples(dataset_name=name) if e.inputs["question"] in selected
    ]
    experiment = await aevaluate(
        replay,
        data=examples,
        evaluators=[_score(k) for k in ("pass", "match_relaxed", "match_strict", "safe")],
        experiment_prefix=meta["label"],
        metadata={k: meta[k] for k in ("model", "commit", "config", "match_rule")},
        client=client,
        max_concurrency=4,
    )
    return getattr(experiment, "url", None) or experiment.experiment_name
