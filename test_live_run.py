import os
import sys
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

from faultshield.faults.scenarios import BENCHMARK_SCENARIOS
from faultshield.agent.nemotron import NemotronRecoveryAgent
from faultshield.benchmark.runner import BenchmarkRunner

print("=" * 70)
print("TESTING LIVE FAULTSHIELD EXECUTION WITH NEBIUS TOKEN FACTORY KEY")
print("=" * 70)

api_key = os.environ.get("NEBIUS_API_KEY", "").strip()
base_url = os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/").strip()
model = os.environ.get("NEMOTRON_MODEL", "nvidia/nemotron-3-super-120b-a12b").strip()

print(f"API Key present: {bool(api_key)}")
print(f"Key preview: {api_key[:15]}...{api_key[-6:]}" if api_key else "None")
print(f"Base URL: {base_url}")
print(f"Target Model: {model}\n")

agent = NemotronRecoveryAgent(
    api_key=api_key,
    base_url=base_url,
    model_name=model,
    use_mock=False,  # Live call requested
)

runner = BenchmarkRunner(nemotron_agent=agent)
scenario = BENCHMARK_SCENARIOS["exp2_frozen_lidar"]

print(f"Running scenario: {scenario.name} (Variant D: FaultShield)...")
res = runner.run_episode(scenario, variant_id="Variant_D", seed=42)
m = res.metrics

print("\n" + "=" * 70)
print("EXPERIMENT RESULTS:")
print("=" * 70)
print(f"Outcome: {'COLLIDED' if m.collision_occurred else ('GOAL REACHED' if m.goal_reached else 'SAFE HALT / RECOVERED')}")
print(f"Duration: {m.duration_sec:.2f} s")
print(f"Path Length: {m.path_length_meters:.2f} m")
print(f"Min Obstacle Clearance: {m.min_obstacle_clearance_meters:.2f} m")
print(f"Stale Decisions Blocked: {m.stale_decisions_rejected}")

if res.ai_decisions:
    dec = res.ai_decisions[0]
    print("\n--- AI RECOVERY DECISION ---")
    print(f"Mode: {'LOCAL FALLBACK (Offline Policy)' if dec.is_simulated else '🟢 LIVE Token Factory call'}")
    print(f"Decision ID: {dec.decision_id}")
    print(f"Recommended Action: {dec.recommended_action}")
    print(f"Confidence: {dec.confidence:.2f}")
    print(f"Diagnosis: {dec.root_cause_diagnosis}")
    print(f"Safety Rationale: {dec.safety_rationale}")
    print(f"Latency: {dec.latency_sec:.3f} s")
    print(f"Raw response: {dec.raw_response}")
else:
    print("\nNo AI decisions were triggered.")

print("\n--- SAFETY GOVERNOR LOG (NON-PASS EVENTS) ---")
for g in res.governor_log:
    if g.status != "NORMAL_PASS":
        print(f" - [{g.status}] v={g.executed_v:.2f}, omega={g.executed_omega:.2f} | Reason: {g.reason}")

print("=" * 70)
