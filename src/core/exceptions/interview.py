"""Interview agent, plan, and question-generation exceptions."""

from src.core.exceptions.base import InternalServerException, NotFoundException


class InterviewConfigurationException(InternalServerException):
    message = "Interview configuration is invalid."
    error_code = "INTERVIEW_CONFIGURATION_ERROR"


class InterviewContextNotFoundException(NotFoundException):
    message = "Candidate assessment context not found."
    error_code = "INTERVIEW_CONTEXT_NOT_FOUND"


class InterviewPlanMissingException(InterviewConfigurationException):
    message = "Interview plan is missing."
    error_code = "INTERVIEW_PLAN_MISSING"


class InterviewPlanInvalidException(InterviewConfigurationException):
    message = "Interview plan is invalid."
    error_code = "INTERVIEW_PLAN_INVALID"


class QuestionValidationException(ValueError):
    """Raised when generated question output fails quality checks."""

    error_code = "INTERVIEW_QUESTION_VALIDATION_FAILED"


class QuestionGeneratorContractException(QuestionValidationException):
    error_code = "INTERVIEW_QUESTION_GENERATOR_CONTRACT_FAILED"


class QuestionGenerationFailedException(InternalServerException):
    message = "Unable to generate a valid interview question."
    error_code = "INTERVIEW_QUESTION_GENERATION_FAILED"


class AgentRuntimeException(InternalServerException):
    message = "Interview agent runtime error."
    error_code = "INTERVIEW_AGENT_RUNTIME_ERROR"


class UnsupportedSessionModeException(AgentRuntimeException):
    message = "Unsupported LiveKit session mode."
    error_code = "INTERVIEW_UNSUPPORTED_SESSION_MODE"


class MissingAgentMetadataException(AgentRuntimeException):
    message = "Required LiveKit agent metadata is missing."
    error_code = "INTERVIEW_MISSING_AGENT_METADATA"
