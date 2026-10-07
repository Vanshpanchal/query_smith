"""Structured outputs the LLM must return (validated by `with_structured_output`)."""

from typing import Literal

from pydantic import BaseModel, Field


class StandaloneQuestion(BaseModel):
    question: str = Field(
        description="The user's latest request rewritten as one self-contained question that "
        "can be understood without the conversation"
    )


class Intent(BaseModel):
    intent: Literal["sql", "clarify", "decline"] = Field(
        description="'sql' if the question can be answered from this dataset with a SELECT; "
        "'clarify' if it has several reasonable readings that would give different answers; "
        "'decline' if it is off-topic, asks to change data, or asks about the system itself"
    )
    reason: str = Field(description="One short sentence explaining the choice")
    readings: list[str] = Field(
        default_factory=list,
        description="Only for 'clarify': 2-3 different meanings of the question, each a short "
        "phrase that completes 'Do you mean ...?' in terms of this dataset",
    )
    clarifying_question: str | None = Field(
        default=None,
        description="Only for 'clarify': the full question to ask the user, ending with '?'",
    )
    assumption: str | None = Field(
        default=None,
        description="Only for 'sql' when the question has more than one reasonable reading: "
        "the reading you chose, in one short sentence",
    )


class SqlDraft(BaseModel):
    reasoning: str = Field(description="1-3 sentences: tables, joins and filters chosen, and why")
    sql: str = Field(description="One PostgreSQL SELECT statement, no trailing semicolon")
    tables_used: list[str] = Field(default_factory=list, description="Table names the query uses")


class FinalAnswer(BaseModel):
    answer: str = Field(
        description="Direct answer in 1-3 sentences using only values present in the rows"
    )
