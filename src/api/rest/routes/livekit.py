"""LiveKit endpoints for candidate interview sessions."""

from __future__ import annotations

from fastapi import APIRouter, Depends, status

from src.core.services.core_api_session_service import CoreApiSessionService
from src.core.services.livekit_token_service import LiveKitTokenService
from src.schemas.common import APIResponse
from src.schemas.livekit import (
    CandidateSessionBootstrapResponse,
    CandidateSessionContextRequest,
    CandidateSessionEntryRequest,
    LiveKitTokenRequest,
    LiveKitTokenResponse,
)

router = APIRouter(prefix="/livekit", tags=["livekit"])


def get_livekit_token_service() -> LiveKitTokenService:
    """Build the LiveKit token service for the request."""

    return LiveKitTokenService()


def get_core_api_session_service() -> CoreApiSessionService:
    """Build the core-api session lifecycle delegate."""

    return CoreApiSessionService()


@router.post(
    "/session-entry",
    response_model=APIResponse[CandidateSessionBootstrapResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Exchange a one-time invitation for a candidate session token",
    description=(
        "Consume a valid candidate invitation or restore its existing active "
        "browser session credential."
    ),
)
async def enter_candidate_session(
    payload: CandidateSessionEntryRequest,
    service: CoreApiSessionService = Depends(get_core_api_session_service),
) -> APIResponse[CandidateSessionBootstrapResponse]:
    """Exchange an invitation for durable candidate session context."""

    session = await service.enter_with_invitation(payload.invitation_token)
    return APIResponse(
        message=(
            "Existing interview session restored."
            if session.invite_reissued
            else "Interview invitation consumed successfully."
        ),
        data=session,
    )


@router.post(
    "/session-context",
    response_model=APIResponse[CandidateSessionBootstrapResponse],
    summary="Restore waiting-room context from a candidate session token",
    description=(
        "Validate the browser session credential and reload candidate, "
        "assessment, timer, and reconnection context."
    ),
)
async def get_candidate_session_context(
    payload: CandidateSessionContextRequest,
    service: CoreApiSessionService = Depends(get_core_api_session_service),
) -> APIResponse[CandidateSessionBootstrapResponse]:
    """Restore candidate session context from a browser session token."""

    session = await service.get_session_context(payload.session_token)
    return APIResponse(
        message="Interview session restored successfully.",
        data=session,
    )


@router.post(
    "/candidate-token",
    response_model=APIResponse[LiveKitTokenResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Create candidate LiveKit token",
    description="Validate an invitation token and issue room credentials for the LiveKit voice interview.",
)
async def create_candidate_token(
    payload: LiveKitTokenRequest,
    service: LiveKitTokenService = Depends(get_livekit_token_service),
) -> APIResponse[LiveKitTokenResponse]:
    """Return signed LiveKit credentials for a candidate interview room."""

    token = await service.create_candidate_token(payload)
    return APIResponse(
        message="LiveKit interview token created successfully.",
        data=token,
    )


@router.post(
    "/demo-token",
    response_model=APIResponse[LiveKitTokenResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Create candidate LiveKit demo token",
    description=(
        "Validate an invitation token and issue credentials for an isolated "
        "static-response practice room."
    ),
)
async def create_demo_token(
    payload: LiveKitTokenRequest,
    service: LiveKitTokenService = Depends(get_livekit_token_service),
) -> APIResponse[LiveKitTokenResponse]:
    """Return signed LiveKit credentials for the no-LLM demo interview."""

    token = await service.create_demo_token(payload)
    return APIResponse(
        message="LiveKit demo token created successfully.",
        data=token,
    )
