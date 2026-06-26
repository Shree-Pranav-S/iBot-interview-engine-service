"""LiveKit endpoints for candidate interview sessions."""

from __future__ import annotations

from fastapi import APIRouter, Depends, status

from src.core.services.livekit_token_service import LiveKitTokenService
from src.schemas.common import APIResponse
from src.schemas.livekit import LiveKitTokenRequest, LiveKitTokenResponse

router = APIRouter(prefix="/livekit", tags=["livekit"])


def get_livekit_token_service() -> LiveKitTokenService:
    """Build the LiveKit token service for the request."""

    return LiveKitTokenService()


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
