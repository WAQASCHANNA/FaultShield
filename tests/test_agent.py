import pytest
from faultshield.agent.nemotron import NemotronRecoveryAgent
from faultshield.integrity.monitor import IncidentReport, AnomalyAlert
from faultshield.config import PERMITTED_RECOVERY_ACTIONS


def test_nemotron_mock_policy_decision():
    agent = NemotronRecoveryAgent(use_mock=True)

    incident = IncidentReport(
        incident_id="INC-TEST",
        timestamp=2.5,
        active_anomalies=[
            AnomalyAlert(
                sensor_name="LIDAR",
                anomaly_type="LIDAR_FROZEN_FRAME",
                confidence=0.98,
                discrepancy_value=0.0,
                threshold=1e-4,
                message="Frozen frame",
            )
        ],
        primary_fault="LIDAR_FROZEN_FRAME",
        risk_level="CRITICAL",
        perceived_robot_state={"odom_x": 3.0, "odom_y": 3.0},
        raw_discrepancy_details="Frozen LiDAR",
    )

    decision = agent.request_recovery(incident, sim_time=2.5, artificial_delay_sec=0.2)
    assert decision.recommended_action in PERMITTED_RECOVERY_ACTIONS
    assert decision.recommended_action == "REVERSE_SLOWLY"
    assert decision.confidence > 0.90
    assert decision.arrival_timestamp > 2.5
