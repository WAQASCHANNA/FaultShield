"""Optional Nemotron integration through Nebius Token Factory."""
from __future__ import annotations

import json
import os
import re
from typing import Any

ALLOWED_ACTIONS = {"BYPASS_LEFT", "BYPASS_RIGHT", "SAFE_STOP", "CONTINUE"}


def has_nebius_key() -> bool:
    return bool(os.getenv("NEBIUS_API_KEY", "").strip())


def request_recovery_recommendation(scenario: str, model_override: str | None = None) -> dict[str, Any]:
    """Call Token Factory at runtime and parse a constrained action response.

    Raises RuntimeError if credentials or API access are unavailable. The UI
    intentionally uses a clearly labelled local fallback instead of pretending
    a live model response occurred.
    """
    api_key = os.getenv("NEBIUS_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("NEBIUS_API_KEY is not configured.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Missing dependency 'openai'. Run: pip install -r requirements.txt") from exc

    base_url = os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")
    model = (
        model_override
        or os.getenv("NEBIUS_MODEL")
        or os.getenv("NEMOTRON_MODEL")
        or "nvidia/nemotron-3-super-120b-a12b"
    )
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=25.0, max_retries=1)

    system_prompt = (
        "You are a high-level decision assistant in a VIRTUAL mobile-robot simulation. "
        "Never issue motor-level commands. Select exactly one action from: "
        "BYPASS_LEFT, BYPASS_RIGHT, SAFE_STOP, CONTINUE. "
        "If sensor integrity is uncertain, do not choose CONTINUE. "
        "Return only valid JSON with keys action, suspected_fault, rationale. "
        "The action is only a recommendation; an independent safety governor validates it."
    )
    user_payload = {
        "scenario": scenario,
        "robot_state": "Mobile robot approaching a known virtual obstacle.",
        "observed_incident": (
            "LiDAR range telemetry became stale or delayed. The robot's local monitor has detected an integrity problem."
            if scenario != "Healthy sensors"
            else "No fault is injected. Continue routine navigation."
        ),
        "permitted_actions": sorted(ALLOWED_ACTIONS),
        "safety_rule": "If continuing could rely on stale sensor data, choose a detour or safe stop.",
    }

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
            temperature=0.1,
            max_tokens=1200,
        )
        content = response.choices[0].message.content or ""
    except Exception as exc:  # surfaced to UI; local simulator remains usable
        raise RuntimeError(f"Token Factory request failed: {exc}") from exc

    parsed: dict[str, Any] | None = None
    try:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if match:
            value = json.loads(match.group(0))
            if isinstance(value, dict):
                parsed = value
    except (json.JSONDecodeError, TypeError):
        parsed = None

    action = str((parsed or {}).get("action", "")).strip().upper()
    if action not in ALLOWED_ACTIONS:
        # Keep the raw answer available for diagnosis, but do not accept an
        # unrecognized action as a controller command.
        action = "BYPASS_LEFT"
        parse_warning = "Model output was not valid constrained JSON; local safe fallback selected."
    else:
        parse_warning = None

    return {
        "live_model_call": True,
        "model": model,
        "action": action,
        "suspected_fault": str((parsed or {}).get("suspected_fault", "Not parsed")),
        "rationale": str((parsed or {}).get("rationale", content[:500])),
        "raw_response": content,
        "parse_warning": parse_warning,
    }

