"""Structured outputs the LLM must return (validated by `with_structured_output`)."""

from typing import Literal

from pydantic import BaseModel, Field


class Intent(BaseModel):
    # Phase 3 adds "clarify" (+ clarifying_question) for ambiguous questions
    intent: Literal["sql", "decline"] = Field(
        description="'sql' if the question can be answered from this dataset with a SELECT; "
        "'decline' if it is off-topic, asks to change data, or asks about the system itself"
    )
    reason: str = Field(description="One short sentence explaining the choice")


class SqlDraft(BaseModel):
    reasoning: str = Field(description="1-3 sentences: tables, joins and filters chosen, and why")
    sql: str = Field(description="One PostgreSQL SELECT statement, no trailing semicolon")
    tables_used: list[str] = Field(default_factory=list, description="Table names the query uses")


class FinalAnswer(BaseModel):
    answer: str = Field(
        description="Direct answer in 1-3 sentences using only values present in the rows"
    )
