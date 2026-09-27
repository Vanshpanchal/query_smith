# AskData

Ask questions about any dataset you load, in plain English, and get a correct, verified answer.
A LangGraph agent picks the relevant tables, writes PostgreSQL, checks it with a SQL parser and
`EXPLAIN`, runs it read-only, repairs itself on failure (max 3 attempts), asks when a question is
ambiguous, and returns the answer + the SQL used + a chart spec.

Full design docs live in the Second Brain vault — see `AI_CONTEXT.md` for the pointer.

## Status

Phase 0 (setup) — repo scaffolded, Postgres + Redis run locally via Docker, app tables migrated.
No agent code yet; see the vault's `Tasks.md` for the build plan.

## Quickstart

```bash
cp .env.example .env            # fill LLM_MODEL / API keys / passwords
make up                         # postgres + redis
make migrate                    # app schema (datasets, threads, schema_cards, query_log, feedback)
make dev                        # uv run uvicorn ask_data.main:app --reload
curl localhost:8000/health
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
