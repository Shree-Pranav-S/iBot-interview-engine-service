"""Pydantic schemas for LLM responses used by interview prompts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
    "question_about_question",
    "skip_question",
]


class StrictPromptModel(BaseModel):
    """Base model that rejects unexpected LLM response keys."""

    model_config = ConfigDict(extra="forbid")


class CandidateResponseClassification(StrictPromptModel):
    response_type: ResponseType
    candidate_question_intent: CandidateQuestionIntent | None = None
    resume_skill_match: bool = False
    is_substantial: bool = False


class QuestionGenerationResponse(StrictPromptModel):
    question_text: str = Field(min_length=12, max_length=500)
    difficulty: Difficulty
    expected_signals: list[str] = Field(default_factory=list, max_length=5)


class AnswerEvaluationResponse(StrictPromptModel):
    strength: Literal["weak", "adequate", "strong"]


LiveResponseType = Literal["answer", "clarification", "irrelevant"]
ClarificationType = Literal[
    "repeat_question",
    "rephrase_question",
    "skip_question",
    "question_doubt",
    "time_to_think",
]
AnswerStrength = Literal["weak", "adequate", "strong"]
LiveNextAction = Literal[
    "ask_next_question",
    "ask_elaboration",
    "repeat_current_question",
    "rephrase_current_question",
    "handle_question_doubt",
    "skip_current_question",
    "start_think_timer",
    "redirect_irrelevant",
]


class LiveInterviewDecision(StrictPromptModel):
    response_type: LiveResponseType
    clarification_type: ClarificationType | None = None
    is_substantial: bool | None = None
    answer_strength: AnswerStrength | None = None
    next_action: LiveNextAction
    interviewer_text: str | None = Field(default=None, max_length=450)
    next_question_text: str | None = Field(default=None, max_length=350)
    next_difficulty: Difficulty | None = None
    expected_signals: list[str] = Field(default_factory=list, max_length=5)
    reason: str = Field(default="", max_length=300)

    @model_validator(mode="after")
    def validate_decision(self) -> LiveInterviewDecision:
        if self.response_type == "answer":
            if self.clarification_type is not None:
                raise ValueError("clarification_type must be null for answer")
            if self.is_substantial is None:
                raise ValueError("is_substantial is required for answer")
            if self.is_substantial:
                if self.answer_strength is None:
                    raise ValueError("answer_strength is required")
                if self.next_action != "ask_next_question":
                    raise ValueError("substantial answers must ask_next_question")
                if not self.next_question_text:
                    raise ValueError("next_question_text is required")
            elif self.next_action != "ask_elaboration":
                raise ValueError("non-substantial answers must ask_elaboration")

        if self.response_type == "clarification":
            if self.clarification_type is None:
                raise ValueError("clarification_type is required")
            if self.is_substantial is not None or self.answer_strength is not None:
                raise ValueError("answer fields must be null for clarification")
            if self.next_action not in {
                "repeat_current_question",
                "rephrase_current_question",
                "handle_question_doubt",
                "skip_current_question",
                "start_think_timer",
            }:
                raise ValueError("invalid clarification next_action")

        if self.response_type == "irrelevant":
            if self.clarification_type is not None:
                raise ValueError("clarification_type must be null for irrelevant")
            if self.is_substantial is not None or self.answer_strength is not None:
                raise ValueError("answer fields must be null for irrelevant")
            if self.next_action != "redirect_irrelevant":
                raise ValueError("irrelevant responses must redirect_irrelevant")

        return self
