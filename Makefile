
.PHONY: up migrate load data dev test redteam eval experiments stack lint reset-db

up:
	docker compose up -d postgres redis

migrate:
	uv run python scripts/migrate.py

load:
	uv run python scripts/load_dataset.py $(DATASET)
	uv run python scripts/index_schema.py $(DATASET)

data: migrate
	uv run python scripts/fetch_ipl.py
	$(MAKE) load DATASET=ipl

dev:
	uv run uvicorn ask_data.main:app --reload

test:
	uv run pytest -q

redteam:
	uv run pytest tests/redteam -q

eval:  # make eval DATASET=ipl [ARGS="--attempts 1 --label e1-attempts-1"]
	uv run python evals/run_evals.py --dataset $(DATASET) --split test $(ARGS)

experiments:  # baseline + E1-E4 on ipl and sales; resumable, stops at the provider's rate limit
	uv run python evals/experiments.py $(ARGS)

stack:  # everything in docker: postgres, redis, api (migrates + loads AUTOLOAD_DATASETS on start)
	docker compose up -d --build

lint:
	uv run ruff check . && uv run ruff format --check .

reset-db:
	docker compose down -v
	$(MAKE) up
	$(MAKE) data
