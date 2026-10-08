#!/bin/sh
# Container entrypoint: migrate + load missing datasets, then serve.
set -e
python scripts/bootstrap.py
exec uvicorn ask_data.main:app --host 0.0.0.0 --port "${PORT:-8000}" --proxy-headers
