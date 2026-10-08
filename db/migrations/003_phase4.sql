-- Phase 4: answer cache. A cached answer is still one query_log row (feedback needs its id), but
-- it ran no LLM and no SQL, so latency/accuracy reports must be able to leave it out.
ALTER TABLE app.query_log ADD COLUMN cached boolean NOT NULL DEFAULT false;
