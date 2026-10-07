-- Phase 3: schema retrieval + query log.

-- A card's vector is only comparable with a query vector from the same embedding model
-- (dimensions and spaces differ), so the model is stored with the card and searched by it.
ALTER TABLE app.schema_cards ADD COLUMN embedding_model text NOT NULL DEFAULT '';
ALTER TABLE app.schema_cards ALTER COLUMN embedding_model DROP DEFAULT;

-- The thread view and "last run of this thread" lookups
CREATE INDEX ON app.query_log (thread_id, created_at DESC);

-- Feedback can be changed (👍 → 👎): one row per query, updated in place
ALTER TABLE app.feedback ADD COLUMN updated_at timestamptz NOT NULL DEFAULT now();
