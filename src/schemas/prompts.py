"""Strict Pydantic contracts for interview-workflow LLM calls."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictPromptModel(BaseModel):
    """Reject coercion and unexpected keys from model-produced JSON."""

    model_config = ConfigDict(extra="forbid", strict=True)


class CandidateResponseClassification(StrictPromptModel):
    """The only accepted output from the response-classification model."""

    response_type: Literal["answer", "clarification", "irrelevant"]
    clarification_type: (
        Literal[
            "repeat_question",
            "rephrase_question",
            "skip_question",
            "question_doubt",
            "time_to_think",
        ]
        | None
    )
    is_substantial: bool | None
    question_doubt_response: str | None = Field(max_length=420)

    @model_validator(mode="after")
    def validate_classification(self) -> CandidateResponseClassification:
        if self.response_type == "answer":
            if self.clarification_type is not None:
                raise ValueError("answer requires clarification_type=null")
            if self.is_substantial is None:
                raise ValueError("answer requires is_substantial")
            if self.question_doubt_response is not None:
                raise ValueError("answer requires question_doubt_response=null")
            return self

        if self.response_type == "clarification":
            if self.clarification_type is None:
                raise ValueError("clarification requires clarification_type")
            if self.is_substantial is not None:
                raise ValueError("clarification requires is_substantial=null")
            if (
                self.clarification_type == "question_doubt"
                and not self.question_doubt_response
            ):
                raise ValueError(
                    "question_doubt requires a concise question_doubt_response"
                )
            if (
                self.clarification_type != "question_doubt"
                and self.question_doubt_response is not None
            ):
                raise ValueError(
                    "question_doubt_response is only valid for question_doubt"
                )
            return self

        if self.clarification_type is not None:
            raise ValueError("irrelevant requires clarification_type=null")
        if self.is_substantial is not None:
            raise ValueError("irrelevant requires is_substantial=null")
        if self.question_doubt_response is not None:
            raise ValueError("irrelevant requires question_doubt_response=null")
        return self


class AnswerEvaluationResponse(StrictPromptModel):
    """The only accepted output from the live answer-evaluation model."""

    strength: Literal["weak", "adequate", "strong"]


FORBIDDEN_INTERVIEWER_FEEDBACK = (
    "good answer",
    "correct",
    "that's perfect",
    "that is perfect",
    "exactly right",
    "great job",
    "well done",
)


def _spoken_word_count(value: str) -> int:
    return len(re.findall(r"\b[\w+#./'-]+\b", value))


class QuestionRephraseResponse(StrictPromptModel):
    """A genuinely rewritten version of the active interview question."""

    question_text: str = Field(min_length=12, max_length=240)

    @model_validator(mode="after")
    def validate_rephrased_question(self) -> QuestionRephraseResponse:
        if "?" not in self.question_text:
            raise ValueError("question_text must contain at least one question")
        if _spoken_word_count(self.question_text) > 60:
            raise ValueError("rephrased question must be at most 60 words")
        return self


class TechnicalQuestionGenerationResponse(StrictPromptModel):
    """Strict output for a technical acknowledgement and next question."""

    acknowledgement: str = Field(min_length=1, max_length=160)
    question_text: str = Field(min_length=12, max_length=240)
    difficulty: Literal["easy", "medium", "hard"]
    probe_deeper: bool
    topic: str = Field(min_length=2, max_length=100)

    @model_validator(mode="after")
    def validate_technical_question(
        self,
    ) -> TechnicalQuestionGenerationResponse:
        acknowledgement = self.acknowledgement.casefold()
        if any(phrase in acknowledgement for phrase in FORBIDDEN_INTERVIEWER_FEEDBACK):
            raise ValueError("interviewer response contains evaluative feedback")
        if "?" in self.acknowledgement:
            raise ValueError("acknowledgement must not contain a question")
        if "?" not in self.question_text:
            raise ValueError("question_text must contain at least one question")
        if _spoken_word_count(self.acknowledgement) > 40:
            raise ValueError("acknowledgement must be at most 40 words")
        if _spoken_word_count(self.question_text) > 60:
            raise ValueError("question_text must be at most 60 words")
        return self


class BehaviouralQuestionGenerationResponse(StrictPromptModel):
    """Strict output for the simpler behavioural/cultural generator."""

    acknowledgement: str = Field(min_length=1, max_length=160)
    question_text: str = Field(min_length=12, max_length=240)
    signal_focus: str = Field(min_length=2, max_length=120)

    @model_validator(mode="after")
    def validate_behavioural_question(
        self,
    ) -> BehaviouralQuestionGenerationResponse:
        acknowledgement = self.acknowledgement.casefold()
        if any(phrase in acknowledgement for phrase in FORBIDDEN_INTERVIEWER_FEEDBACK):
            raise ValueError("interviewer response contains evaluative feedback")
        if "?" in self.acknowledgement:
            raise ValueError("acknowledgement must not contain a question")
        if "?" not in self.question_text:
            raise ValueError("question_text must contain at least one question")
        if _spoken_word_count(self.acknowledgement) > 40:
            raise ValueError("acknowledgement must be at most 40 words")
        if _spoken_word_count(self.question_text) > 60:
            raise ValueError("question_text must be at most 60 words")
        return self
