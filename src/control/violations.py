"""Violation and acknowledgement helpers."""

from __future__ import annotations

import random


def build_violation_ack_text(eval_result: dict) -> str:
    detail = str(eval_result.get("contradiction_detail") or "").strip()
    if eval_result.get("resume_contradiction"):
        reference = detail or "a skill listed in your profile"
        return f"Noted, your profile indicates {reference}. Let's continue."
    if eval_result.get("yoe_contradiction"):
        return (
            "I noticed a small discrepancy in the experience you mentioned "
            "versus your profile. We'll note that and continue."
        )
    return ""


def build_evaluation_ack(eval_result: dict) -> str:
    violation_text = build_violation_ack_text(eval_result)
    if violation_text:
        return violation_text
    return random.choice(
        [
            "Got it, thank you.",
            "Understood, let's keep going.",
            "Thanks for that.",
            "Alright, moving on.",
            "Good, let's continue.",
        ]
    )
