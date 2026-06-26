"""Pydantic schemas for LLM responses used by interview prompts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Difficulty = Literal["easy", "medium", "hard"]
ResponseType = Literal[
    "answer",
    "clarification_question",
    "irrelevant_answer",
    "silence",
]
CandidateQuestionIntent = Literal[
    "repeat_question",
    "rephrase_question",
    "skip_question",
]


class StrictPromptModel(BaseModel):
    """Base model that rejects unexpected LLM response keys."""

    model_config = ConfigDict(extra="forbid")


class CandidateResponseClassification(StrictPromptModel):
    response_type: ResponseType
    candidate_question_intent: CandidateQuestionIntent | None = None
    resume_skill_match: bool = False


class QuestionGenerationResponse(StrictPromptModel):
    question_text: str = Field(min_length=12, max_length=500)
    difficulty: Difficulty
    expected_signals: list[str] = Field(default_factory=list, max_length=5)
    rationale: str | None = Field(default=None, max_length=300)


class AnswerEvaluationResponse(StrictPromptModel):
    strength: Literal["weak", "adequate", "strong"]
