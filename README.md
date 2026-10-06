# AskData

Ask questions about any dataset you load, in plain English, and get a correct, verified answer.
A LangGraph agent picks the relevant tables, writes PostgreSQL, checks it with a SQL parser and
`EXPLAIN`, runs it read-only, repairs itself on failure (max 3 attempts), asks when a question is
ambiguous, and returns the answer + the SQL used + a chart spec.

Full design docs live in the Second Brain vault — see `AI_CONTEXT.md` for the pointer.

## Status

Phase 2 (v1) — self-correcting, guarded API. `classify → select_tables → generate_sql →
validate_sql (sqlglot guard + EXPLAIN cost cap) → execute_sql (read-only role) → check_result →
answer`, with every check looping back to `generate_sql` (max 3 attempts) or to `give_up`.
Works unchanged on every loaded dataset (`ipl`, `sales`, …). Next: Phase 3 (follow-ups,
clarification, schema retrieval). Build plan: the vault's `Tasks.md`.

## Quickstart

```bash
cp .env.example .env            # fill LLM_MODEL + provider key, passwords, API_KEYS
make up                         # postgres + redis
make data                       # migrate + fetch + load ipl
make load DATASET=sales         # any folder in datasets/
make dev                        # uv run uvicorn ask_data.main:app --reload

curl localhost:8000/ready
TID=$(curl -s -X POST localhost:8000/api/v1/threads -H "X-API-Key: dev-key" \
  -H "Content-Type: application/json" -d '{"dataset":"ipl"}' | jq -r .thread_id)
curl -N -X POST localhost:8000/api/v1/threads/$TID/ask -H "X-API-Key: dev-key" \
  -H "Content-Type: application/json" -d '{"question":"Top 5 run scorers of all time"}'
```

SSE events: `status`, `tables`, `sql`, `retry`, `rows`, `answer`, `error`, `done`.
Non-streaming: `POST /api/v1/ask {"question", "dataset"?, "thread_id"?}`.

## Tests

```bash
make test       # unit + graph (scripted LLM) + red-team + integration (needs `make up`)
make redteam    # 12 attacks, each blocked by the guard AND by the database role
```

## Repo layout

```
db/            role setup (init) + numbered SQL migrations
datasets/      one folder per dataset (dataset.yaml + notes + fewshots + golden set)
scripts/       loader, schema indexer, migration runner
src/ask_data/  FastAPI app + LangGraph graph (nodes, routing, state)
evals/         execution-accuracy harness against golden.yaml
tests/         unit / graph / integration / redteam
```
