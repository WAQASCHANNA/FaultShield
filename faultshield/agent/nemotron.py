"""
NVIDIA Nemotron Recovery Client via Nebius Token Factory.
Supports:
1. Live runtime API calls to Nebius Token Factory endpoint
2. Intelligent mock / offline policy with simulated inference latency
3. Robust error handling and JSON validation
"""

import os
import json
import time
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any
from faultshield.config import (
    DEFAULT_NEBIUS_BASE_URL,
    DEFAULT_NEMOTRON_MODEL,
    PERMITTED_RECOVERY_ACTIONS,
    DEFAULT_REQUEST_TIMEOUT,
)
from faultshield.integrity.monitor import IncidentReport
from faultshield.agent.prompts import SYSTEM_PROMPT, build_incident_prompt


@dataclass
class NemotronDecision:
    decision_id: str
    incident_id: str
    recommended_action: str
    target_linear_velocity: float
    target_angular_velocity: float
    confidence: float
    root_cause_diagnosis: str
    safety_rationale: str
    requires_immediate_stop: bool
    dispatch_timestamp: float
    arrival_timestamp: float
    latency_sec: float
    is_simulated: bool
    raw_response: str
    error_message: Optional[str] = None


LEGACY_MODEL_MAP = {
    "nvidia/nemotron-4-340b-instruct": DEFAULT_NEMOTRON_MODEL,
    "nvidia/llama-3.1-nemotron-70b-instruct": DEFAULT_NEMOTRON_MODEL,
}


class NemotronRecoveryAgent:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_NEBIUS_BASE_URL,
        model_name: str = DEFAULT_NEMOTRON_MODEL,
        use_mock: bool = False,
    ):
        self.api_key = api_key if api_key is not None else os.environ.get("NEBIUS_API_KEY", "")
        self.base_url = base_url or os.environ.get("NEBIUS_BASE_URL", DEFAULT_NEBIUS_BASE_URL)
        self.model_name = LEGACY_MODEL_MAP.get(model_name, model_name)
        self.use_mock = use_mock or (not self.api_key.strip())
        self._openai_client = None
        self._live_cache: Dict[str, Dict[str, Any]] = {}

        if not self.use_mock and self.api_key.strip():
            try:
                from openai import OpenAI
                self._openai_client = OpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key.strip(),
                    timeout=DEFAULT_REQUEST_TIMEOUT,
                )
            except Exception as e:
                print(f"[NemotronAgent] Failed to initialize OpenAI client: {e}. Falling back to mock.")
                self.use_mock = True

    def _generate_mock_decision(
        self,
        incident: IncidentReport,
        sim_time: float,
        artificial_delay_sec: float
    ) -> NemotronDecision:
        """Deterministic, highly accurate offline policy matching Nemotron recovery behavior."""
        primary = incident.primary_fault

        if "LIDAR_FROZEN" in primary:
            action = "REVERSE_SLOWLY"
            v = -0.20
            omega = 0.0
            diagnosis = "LiDAR range reading buffer frozen in stale frame; undetected obstacles ahead."
            rationale = "Reversing along recent verified path disengages robot from blind zone and enables buffer flush."
            stop_imm = True
        elif "BLIND" in primary or "DROPOUT" in primary:
            action = "SAFE_STOP_AND_HOLD"
            v = 0.0
            omega = 0.0
            diagnosis = "LiDAR sensor blackout: all beams reporting max saturation."
            rationale = "Total loss of range visibility requires immediate emergency stop."
            stop_imm = True
        elif "IMU" in primary or "ODOMETRY" in primary:
            action = "RECALIBRATE_IMU"
            v = 0.0
            omega = 0.0
            diagnosis = "Kinematic cross-sensor contradiction: IMU yaw rate diverges from wheel odometry."
            rationale = "Temporary standstill allows stationary bias estimation and sensor fusion realignment."
            stop_imm = False
        elif "STALE" in primary:
            action = "SAFE_STOP_AND_HOLD"
            v = 0.0
            omega = 0.0
            diagnosis = "Telemetry bus latency exceeded safety timeout."
            rationale = "Moving on stale telemetry violates physical AI safety envelope."
            stop_imm = True
        else:
            action = "SWITCH_TO_CONSERVATIVE_CONTOURING"
            v = 0.15
            omega = 0.2
            diagnosis = "Minor telemetry variance anomaly detected."
            rationale = "Reduce operational envelope and widen safety buffers."
            stop_imm = False

        # Latency calculation
        mock_internal_latency = 0.08  # ~80ms baseline local compute
        total_latency = mock_internal_latency + artificial_delay_sec
        arrival_t = sim_time + total_latency

        raw = json.dumps({
            "recommended_action": action,
            "target_linear_velocity": v,
            "target_angular_velocity": omega,
            "confidence": 0.95,
            "root_cause_diagnosis": diagnosis,
            "safety_rationale": rationale,
            "requires_immediate_stop": stop_imm,
        }, indent=2)

        return NemotronDecision(
            decision_id=f"DEC-{incident.incident_id}",
            incident_id=incident.incident_id,
            recommended_action=action,
            target_linear_velocity=v,
            target_angular_velocity=omega,
            confidence=0.95,
            root_cause_diagnosis=diagnosis,
            safety_rationale=rationale,
            requires_immediate_stop=stop_imm,
            dispatch_timestamp=sim_time,
            arrival_timestamp=arrival_t,
            latency_sec=total_latency,
            is_simulated=True,
            raw_response=raw,
        )

    def request_recovery(
        self,
        incident: IncidentReport,
        sim_time: float,
        artificial_delay_sec: float = 0.0
    ) -> NemotronDecision:
        """
        Request recovery decision from NVIDIA Nemotron (or fallback).
        Applies artificial_delay_sec to simulate network latency / packet delay.
        """
        if self.use_mock or self._openai_client is None:
            return self._generate_mock_decision(incident, sim_time, artificial_delay_sec)

        # Live call to Nebius Token Factory
        cache_key = f"{self.model_name}:{incident.primary_fault}"
        prompt = build_incident_prompt(incident.to_dict())
        t0 = time.time()
        try:
            if cache_key in self._live_cache:
                cached = self._live_cache[cache_key]
                raw_text = cached["raw_text"]
                parsed = cached["parsed"]
                elapsed_wall = cached["elapsed_wall"]
            else:
                response = self._openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=1200,
                )
                elapsed_wall = time.time() - t0
                raw_text = response.choices[0].message.content or "{}"

                # Clean markdown formatting or reasoning preamble if present
                cleaned = re.sub(r"^```json\s*", "", raw_text.strip(), flags=re.MULTILINE)
                cleaned = re.sub(r"^```\s*", "", cleaned.strip(), flags=re.MULTILINE)
                cleaned = cleaned.rstrip("`").strip()
                json_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
                if json_match:
                    cleaned = json_match.group(0)

                parsed = json.loads(cleaned)
                self._live_cache[cache_key] = {
                    "raw_text": raw_text,
                    "parsed": parsed,
                    "elapsed_wall": elapsed_wall,
                }

            action = parsed.get("recommended_action", "SAFE_STOP_AND_HOLD")
            if action not in PERMITTED_RECOVERY_ACTIONS:
                action = "SAFE_STOP_AND_HOLD"

            sim_latency = 0.08 + artificial_delay_sec
            arrival_t = sim_time + sim_latency

            return NemotronDecision(
                decision_id=f"DEC-{incident.incident_id}",
                incident_id=incident.incident_id,
                recommended_action=action,
                target_linear_velocity=float(parsed.get("target_linear_velocity", 0.0)),
                target_angular_velocity=float(parsed.get("target_angular_velocity", 0.0)),
                confidence=float(parsed.get("confidence", 0.90)),
                root_cause_diagnosis=str(parsed.get("root_cause_diagnosis", "Nebius Nemotron diagnosis")),
                safety_rationale=str(parsed.get("safety_rationale", "Autonomous recovery action")),
                requires_immediate_stop=bool(parsed.get("requires_immediate_stop", True)),
                dispatch_timestamp=sim_time,
                arrival_timestamp=arrival_t,
                latency_sec=round(elapsed_wall + artificial_delay_sec, 3),
                is_simulated=False,
                raw_response=raw_text,
            )
        except Exception as err:
            print(f"[NemotronAgent] API call error: {err}. Falling back to mock decision.")
            mock_dec = self._generate_mock_decision(incident, sim_time, artificial_delay_sec)
            mock_dec.error_message = str(err)
            mock_dec.root_cause_diagnosis += f" (Live API fallback: {str(err)[:60]})"
            return mock_dec
