#!/bin/bash
set -e
psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" <<-SQL
  CREATE EXTENSION IF NOT EXISTS vector;
  CREATE ROLE askdata_app    LOGIN PASSWORD '${APP_DB_PASSWORD}';
  CREATE ROLE askdata_reader LOGIN PASSWORD '${READER_DB_PASSWORD}';

  GRANT ALL ON DATABASE askdata TO askdata_app;
  GRANT ALL ON SCHEMA public TO askdata_app;
  CREATE SCHEMA app AUTHORIZATION askdata_app;
  -- no dataset schemas here: the loader creates them and grants the reader

  -- reader: connect only; SELECT on dataset schemas is granted per dataset by the loader
  REVOKE ALL ON SCHEMA public FROM PUBLIC;
  GRANT CONNECT ON DATABASE askdata TO askdata_reader;
  ALTER ROLE askdata_reader SET default_transaction_read_only = on;
  ALTER ROLE askdata_reader SET statement_timeout = '5s';
  ALTER ROLE askdata_reader SET search_path = '';
  ALTER ROLE askdata_reader CONNECTION LIMIT 10;
SQL
