"""Coverage for NVIDIA NIM streaming transport failures."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.config.settings import settings
from src.core.services.evaluation_errors import TransientEvaluationError
from src.core.services.evaluation_llm_client import _complete


class _BrokenStream:
    def __aiter__(self) -> _BrokenStream:
        return self

    async def __anext__(self) -> Any:
        raise httpx.RemoteProtocolError(
            "peer closed connection without sending complete message body "
            "(incomplete chunked read)"
        )


@pytest.mark.asyncio
async def test_stream_interrupt_falls_back_to_non_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "NVIDIA_NIM_STREAM", True)

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=[
            _BrokenStream(),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))
                ]
            ),
        ]
    )

    with patch(
        "src.core.services.evaluation_llm_client._get_client",
        return_value=mock_client,
    ):
        content = await _complete([{"role": "user", "content": "evaluate"}])

    assert content == '{"ok": true}'
    assert mock_client.chat.completions.create.await_count == 2
    assert (
        mock_client.chat.completions.create.await_args_list[1].kwargs["stream"] is False
    )


@pytest.mark.asyncio
async def test_non_stream_transport_error_is_transient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "NVIDIA_NIM_STREAM", False)

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=httpx.RemoteProtocolError("connection dropped")
    )

    with (
        patch(
            "src.core.services.evaluation_llm_client._get_client",
            return_value=mock_client,
        ),
        pytest.raises(TransientEvaluationError, match="transport error"),
    ):
        await _complete([{"role": "user", "content": "evaluate"}])
