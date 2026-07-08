"""Custom exceptions package init."""

from src.core.exceptions.base import (
    AppException,
    AuthenticationException,
    BadGatewayException,
    BadRequestException,
    ConflictException,
    ForbiddenException,
    InternalServerException,
    NotFoundException,
)
from src.core.exceptions.evaluation import (
    EvaluationError,
    EvaluationNotReadyError,
    EvaluationSchemaError,
    PermanentEvaluationError,
    TransientEvaluationError,
)
from src.core.exceptions.interview import (
    AgentRuntimeException,
    InterviewConfigurationException,
    InterviewContextNotFoundException,
    InterviewPlanInvalidException,
    InterviewPlanMissingException,
    MissingAgentMetadataException,
    QuestionGenerationFailedException,
    QuestionGeneratorContractException,
    QuestionValidationException,
    UnsupportedSessionModeException,
)
from src.core.exceptions.llm import LLMConfigurationException, LLMProviderException
from src.core.exceptions.proxy import (
    CoreApiRequestFailedException,
    CoreApiUnreachableException,
)
from src.core.exceptions.session import (
    LiveKitConfigurationException,
    SessionTokenExpiryInvalidException,
    SessionTokenExpiryMissingException,
)

__all__ = [
    "AgentRuntimeException",
    "AppException",
    "AuthenticationException",
    "BadRequestException",
    "ConflictException",
    "CoreApiRequestFailedException",
    "CoreApiUnreachableException",
    "EvaluationError",
    "EvaluationNotReadyError",
    "EvaluationSchemaError",
    "ForbiddenException",
    "InternalServerException",
    "InterviewConfigurationException",
    "InterviewContextNotFoundException",
    "InterviewPlanInvalidException",
    "InterviewPlanMissingException",
    "LLMConfigurationException",
    "LLMProviderException",
    "LiveKitConfigurationException",
    "MissingAgentMetadataException",
    "BadGatewayException",
    "NotFoundException",
    "PermanentEvaluationError",
    "QuestionGenerationFailedException",
    "QuestionGeneratorContractException",
    "QuestionValidationException",
    "SessionTokenExpiryInvalidException",
    "SessionTokenExpiryMissingException",
    "TransientEvaluationError",
    "UnsupportedSessionModeException",
]
