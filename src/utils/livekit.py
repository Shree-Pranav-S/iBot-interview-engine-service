"""LiveKit helper functions for interview voice sessions."""

from __future__ import annotations


def candidate_room_name(candidate_assessment_id: str) -> str:
    """Return the deterministic LiveKit room name for an interview."""

    return f"interview-{candidate_assessment_id}"


def candidate_participant_identity(candidate_id: str) -> str:
    """Return the deterministic LiveKit identity for a candidate participant."""

    return f"candidate-{candidate_id}"


def chunk_for_tts(text: str, *, max_chars: int = 120) -> list[str]:
    """Split text into small sentence-ish chunks for responsive TTS."""

    clean = " ".join((text or "").split())
    if not clean:
        return []

    chunks: list[str] = []
    current: list[str] = []

    for token in clean.split(" "):
        current.append(token)
        joined = " ".join(current)

        if len(joined) >= max_chars or token.endswith((".", "?", "!")):
            chunks.append(joined + " ")
            current = []

    if current:
        chunks.append(" ".join(current))

    return chunks
