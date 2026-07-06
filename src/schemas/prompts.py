"""Strict Pydantic contracts for interview-workflow LLM calls."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictPromptModel(BaseModel):
    """Reject coercion and unexpected keys from model-produced JSON."""

    model_config = ConfigDict(extra="forbid", strict=True)


class CandidateResponseClassification(StrictPromptModel):
    """Classification-only contract for one candidate utterance."""

    response_type: Literal[
        "answer",
        "clarification",
        "irrelevant",
        "interview_meta",
    ]
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
    interview_meta_type: (
        Literal[
            "general_guidance",
            "time_remaining",
        ]
        | None
    )

    @model_validator(mode="after")
    def validate_classification(self) -> CandidateResponseClassification:
        """Enforce mutually exclusive routing fields."""

        if self.response_type == "answer":
            if self.clarification_type is not None:
                raise ValueError("answer requires clarification_type=null")
            if self.is_substantial is None:
                raise ValueError("answer requires is_substantial")
            if self.interview_meta_type is not None:
                raise ValueError("answer requires interview_meta_type=null")
            return self

        if self.response_type == "clarification":
            if self.clarification_type is None:
                raise ValueError("clarification requires clarification_type")
            if self.is_substantial is not None:
                raise ValueError("clarification requires is_substantial=null")
            if self.interview_meta_type is not None:
                raise ValueError("clarification requires interview_meta_type=null")
            return self

        if self.clarification_type is not None:
            raise ValueError(f"{self.response_type} requires clarification_type=null")
        if self.is_substantial is not None:
            raise ValueError(f"{self.response_type} requires is_substantial=null")
        if self.response_type == "interview_meta":
            if self.interview_meta_type is None:
                raise ValueError("interview_meta requires interview_meta_type")
        elif self.interview_meta_type is not None:
            raise ValueError("irrelevant requires interview_meta_type=null")
        return self


class LiveInterviewerResponse(StrictPromptModel):
    """Evaluation/question response produced only after classification."""

    response_mode: Literal["answer", "question_doubt", "rephrase_question"]
    answer_strength: Literal["weak", "adequate", "strong"] | None
    acknowledgement: str | None = Field(max_length=200)
    question_text: str | None = Field(max_length=260)
    topic: str | None = Field(max_length=100)
    clarification_response: str | None = Field(max_length=440)

    @model_validator(mode="after")
    def validate_live_interviewer_response(self) -> LiveInterviewerResponse:
        """Validate mode-specific fields and spoken-text limits."""

        if self.acknowledgement:
            lowered = self.acknowledgement.casefold()
            if any(phrase in lowered for phrase in FORBIDDEN_INTERVIEWER_FEEDBACK):
                raise ValueError("acknowledgement contains evaluative feedback")
            if "?" in self.acknowledgement:
                raise ValueError("acknowledgement must not contain a question")
        if self.question_text and _spoken_word_count(self.question_text) > 60:
            raise ValueError("question_text must be at most 60 words")

        if self.response_mode == "answer":
            if self.question_text and "?" not in self.question_text:
                raise ValueError("question_text must contain a question")
            if self.clarification_response is not None:
                raise ValueError("answer requires clarification_response=null")
            return self

        if self.answer_strength is not None:
            raise ValueError("clarification response requires answer_strength=null")
        if self.acknowledgement is not None:
            raise ValueError("clarification response requires acknowledgement=null")
        if self.question_text is not None:
            raise ValueError("clarification response requires question_text=null")
        if self.topic is not None:
            raise ValueError("clarification response requires topic=null")
        if not self.clarification_response:
            raise ValueError("clarification response text is required")
        if (
            self.response_mode == "rephrase_question"
            and self.clarification_response
            and "?" not in self.clarification_response
        ):
            raise ValueError("a rephrased question must contain a question mark")
        return self


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
        """Require a concise interrogative rewrite."""

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
        """Validate neutral acknowledgement and technical question shape."""

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
        """Validate neutral acknowledgement and behavioural question shape."""

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
