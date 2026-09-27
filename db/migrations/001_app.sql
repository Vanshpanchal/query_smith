CREATE TABLE app.datasets (
  name         text PRIMARY KEY,                -- = folder name in datasets/
  schema_name  text UNIQUE NOT NULL,
  title        text NOT NULL,
  version      text NOT NULL,                   -- sha256(dataset.yaml + raw files)
  table_count  int NOT NULL,
  row_count    bigint NOT NULL,
  loaded_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE app.threads (
  thread_id   text PRIMARY KEY,               -- = LangGraph thread_id
  dataset     text NOT NULL REFERENCES app.datasets(name),
  created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE app.schema_cards (
  dataset     text NOT NULL REFERENCES app.datasets(name) ON DELETE CASCADE,
  table_name  text NOT NULL,
  card        text NOT NULL,
  embedding   vector(768) NOT NULL,
  updated_at  timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (dataset, table_name)
);

CREATE TABLE app.query_log (
  id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  thread_id    text NOT NULL,
  dataset      text NOT NULL,
  run_id       text,                            -- LangSmith run id
  question     text NOT NULL,
  final_sql    text,
  outcome      text NOT NULL,                   -- answered | clarified | declined | gave_up | error
  attempts     smallint NOT NULL DEFAULT 0,
  row_count    int,
  latency_ms   int NOT NULL,
  model        text NOT NULL,
  created_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ON app.query_log (created_at DESC);

CREATE TABLE app.feedback (
  query_id   uuid PRIMARY KEY REFERENCES app.query_log(id),
  rating     smallint NOT NULL CHECK (rating IN (-1, 1)),
  comment    text,
  created_at timestamptz NOT NULL DEFAULT now()
);
