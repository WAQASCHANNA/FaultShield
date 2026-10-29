"""
FaultShield Interactive Dashboard & Evaluation Platform.
Streamlit application for demonstrating cross-layer reliability benchmarking,
sensor fault injection, NVIDIA Nemotron recovery, and the Independent Safety Governor.
"""

import os
import time
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from faultshield.config import (
    DEFAULT_NEBIUS_BASE_URL,
    DEFAULT_NEMOTRON_MODEL,
    PERMITTED_RECOVERY_ACTIONS,
    GOVERNOR_MAX_DECISION_TTL_SEC,
)
from faultshield.core.world import WarehouseMap
from faultshield.faults.scenarios import BENCHMARK_SCENARIOS, Scenario
from faultshield.agent.nemotron import NemotronRecoveryAgent
from faultshield.benchmark.runner import BenchmarkRunner, EpisodeRunResult
from faultshield.benchmark.metrics import compute_benchmark_summary, EpisodeMetrics
from faultshield.visualization.plotter import (
    create_warehouse_figure,
    create_telemetry_timeline_figure,
    create_variant_comparison_chart,
)
from simulation import run_simulation
from visualization_3d import build_3d_replay, build_3d_warehouse_replay

# Page configuration
st.set_page_config(
    page_title="FaultShield - Embodied AI Reliability Benchmark",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling for premium look & feel
st.markdown("""
<style>
    .main { background-color: #0b0f19; }
    .stMetric {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        color: #f8fafc !important;
    }
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
        color: #cbd5e1 !important;
    }
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
        color: #f8fafc !important;
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-nvidia { background-color: #76b900; color: #000; }
    .badge-nebius { background-color: #3b82f6; color: #fff; }
    .badge-track { background-color: #8b5cf6; color: #fff; }
    .badge-safe { background-color: #10b981; color: #fff; }
    .badge-danger { background-color: #ef4444; color: #fff; }
</style>
""", unsafe_allow_html=True)


# ---------------- SIDEBAR CONTROLS ----------------
st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=64)
st.sidebar.title("FaultShield Engine")
st.sidebar.caption("Cross-Layer Embodied AI Reliability Benchmark")

st.sidebar.markdown("---")
st.sidebar.subheader("1. Experiment Selection")

scenario_options = {
    f"{s.name}": sid for sid, s in BENCHMARK_SCENARIOS.items()
}
selected_scenario_label = st.sidebar.selectbox(
    "Benchmark Scenario",
    list(scenario_options.keys()),
    index=1,  # Default to Frozen LiDAR
)
scenario_id = scenario_options[selected_scenario_label]
selected_scenario = BENCHMARK_SCENARIOS[scenario_id]

st.sidebar.info(f"**Description:** {selected_scenario.description}")

st.sidebar.markdown("---")
st.sidebar.subheader("2. Nebius Token Factory / Nemotron")

nebius_api_key = st.sidebar.text_input(
    "Nebius Token Factory API Key",
    value=os.environ.get("NEBIUS_API_KEY", ""),
    type="password",
    help="Enter your Token Factory API key or promotional grant. If empty, high-fidelity local mock policy is used.",
)

use_live_api = st.sidebar.checkbox(
    "Use live Nemotron via Nebius Token Factory",
    value=bool(nebius_api_key.strip()),
    help="When enabled, real runtime inference requests are sent to NVIDIA Nemotron on Nebius Token Factory.",
)

selected_model = st.sidebar.selectbox(
    "Model",
    [
        "nvidia/nemotron-3-super-120b-a12b",
        "nvidia/Nemotron-3-Ultra-550b-a55b",
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
        "nvidia/Nemotron-3_5-Lightning",
    ],
    index=0,
)

simulated_ai_delay = st.sidebar.slider(
    "Simulated Network / Inference Delay (s)",
    min_value=0.0,
    max_value=3.0,
    value=float(selected_scenario.ai_delay_sec),
    step=0.1,
    help="Test how stale recommendations interact with the Safety Governor's 0.5s TTL.",
)

seed = st.sidebar.number_input("Random Seed", min_value=1, max_value=9999, value=42)

run_button = st.sidebar.button("Run experiment", type="primary", use_container_width=True)
run_benchmark_button = st.sidebar.button("⚖ Run 4-Variant Comparison", type="secondary", use_container_width=True)


# ---------------- AGENT & RUNNER INITIALIZATION ----------------
@st.cache_resource
def get_runner(api_key: str, model: str, use_live: bool):
    agent = NemotronRecoveryAgent(
        api_key=api_key if use_live else "",
        model_name=model,
        use_mock=not use_live,
    )
    return BenchmarkRunner(nemotron_agent=agent)

runner = get_runner(nebius_api_key, selected_model, use_live_api)


# ---------------- HEADER SECTION ----------------
st.markdown("""
# 🛡️ FaultShield: Cross-Layer Reliability Benchmark
<span class="badge-pill badge-track">Physical AI Track</span>
<span class="badge-pill badge-nvidia">NVIDIA Nemotron</span>
<span class="badge-pill badge-nebius">Nebius Token Factory</span>
<span class="badge-pill badge-safe">Zero Physical Hardware</span>
""", unsafe_allow_html=True)

st.markdown(
    "**Core Research Question:** *When an embodied AI agent receives corrupted or contradictory sensor data—"
    "and its own reasoning service may be delayed—can it detect when not to act, enter a safe state, and recover reliably?*"
)

# Active scenario override
custom_scenario = Scenario(
    id=selected_scenario.id,
    name=selected_scenario.name,
    description=selected_scenario.description,
    fault_type=selected_scenario.fault_type,
    start_time=selected_scenario.start_time,
    duration=selected_scenario.duration,
    magnitude=selected_scenario.magnitude,
    ai_delay_sec=simulated_ai_delay,
    start_pos=selected_scenario.start_pos,
    goal_pos=selected_scenario.goal_pos,
)


# ---------------- TABS ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Single Run Inspection",
    "⚖️ 4-Variant Benchmark",
    "📡 Telemetry & Decision Log",
    "📊 Batch Statistical Evaluation",
    "🔬 Research Gap & Architecture",
])


# ---------------- TAB 1: SINGLE RUN INSPECTION ----------------
with tab1:
    variant_to_run = st.radio(
        "Select Variant to Inspect:",
        [
            ("Variant_D", "Variant D: FaultShield (Monitor + Nemotron + Safety Governor)"),
            ("Variant_B", "Variant B: Unmitigated Corrupted Telemetry (Baseline Failure)"),
            ("Variant_C", "Variant C: Conventional Monitor + Fixed Freeze Fallback"),
            ("Variant_A", "Variant A: Healthy Baseline (No Injected Faults)"),
        ],
        format_func=lambda x: x[1],
        index=0,
        horizontal=True,
    )[0]

    # Session state for current episode
    if "current_result" not in st.session_state or run_button:
        with st.spinner("Simulating robot trajectory and running cross-layer pipeline..."):
            result = runner.run_episode(custom_scenario, variant_id=variant_to_run, seed=seed)
            baseline_wh_result = runner.run_episode(custom_scenario, variant_id="Variant_B", seed=seed)
            guarded_wh_result = (
                result if variant_to_run == "Variant_D"
                else runner.run_episode(custom_scenario, variant_id="Variant_D", seed=seed)
            )
            st.session_state.current_result = result
            st.session_state.baseline_wh_result = baseline_wh_result
            st.session_state.guarded_wh_result = guarded_wh_result

    result: EpisodeRunResult = st.session_state.current_result
    baseline_wh_result: EpisodeRunResult = st.session_state.get("baseline_wh_result", result)
    guarded_wh_result: EpisodeRunResult = st.session_state.get("guarded_wh_result", result)
    m = result.metrics

    # Key Metrics Display
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        if m.collision_occurred:
            st.metric("Outcome", "CRASHED", delta="- Collision", delta_color="inverse")
        elif m.goal_reached:
            st.metric("Outcome", "GOAL REACHED", delta="+ Success", delta_color="normal")
        else:
            st.metric("Outcome", "SAFE HALT", delta="Protected", delta_color="off")
    with c2:
        st.metric("Duration", f"{m.duration_sec:.2f} s")
    with c3:
        st.metric("Min Obstacle Clearance", f"{m.min_obstacle_clearance_meters:.2f} m")
    with c4:
        st.metric("Stale Decisions Blocked", m.stale_decisions_rejected)
    with c5:
        st.metric("AI Decision Latency", f"{m.mean_ai_latency_sec:.2f} s" if m.mean_ai_latency_sec > 0 else "N/A")

    # AI Recovery Status & Model Response Indicator
    if result.ai_decisions:
        dec = result.ai_decisions[0]
        if not dec.is_simulated:
            st.success(f"🟢 **LIVE Token Factory call** — The app successfully made a real model inference call to **{selected_model}** via Nebius Token Factory.")
        else:
            if getattr(dec, "error_message", None):
                st.error(f"**Live API Error:** {dec.error_message}")
            st.warning("🟡 **LOCAL FALLBACK — not a model response** (The app used its local recovery fallback. Check your .env settings, model ID and API access before trying again).")

        c_dec1, c_dec2 = st.columns([1, 2])
        with c_dec1:
            st.markdown(f"**Model:** `{selected_model}`")
            st.markdown(f"**Recommendation:** `{dec.recommended_action}`")
            st.markdown(f"**Confidence:** `{dec.confidence:.2f}`")
        with c_dec2:
            st.markdown(f"**Root-Cause Diagnosis:** {dec.root_cause_diagnosis}")
            st.markdown(f"**Safety Rationale:** {dec.safety_rationale}")

        with st.expander("🔍 Raw model response"):
            st.code(dec.raw_response, language="json")
    elif custom_scenario.fault_type.value == "NONE":
        st.info("ℹ️ **Healthy sensors selected:** The current code intentionally skips the AI call when no fault has been injected.")

    # ---------------- 3D ROBOT SIMULATION REPLAY (ABOVE 2D DIAGNOSTICS) ----------------
    st.markdown("---")
    st.subheader("🧊 3D robot simulation replay")
    st.caption(
        "Interactive 3D digital-twin view of the experiment trajectories featuring the robot chassis and wheels, "
        "obstacle, floor grid, both robot trajectories (Baseline vs. FaultShield), and Play, Pause, and timeline controls. "
        "**Important distinction:** this view renders our 2D kinematic simulation in an interactive 3D scene; "
        "it is not yet a full 3D rigid-body physics simulator (such as PyBullet or Webots)."
    )

    # Map current scenario + AI decision to the v2 3D replay simulator
    if custom_scenario.fault_type.value == "NONE":
        sim_scenario_name = "Healthy sensors"
    elif custom_scenario.ai_delay_sec >= 0.5 or custom_scenario.fault_type.value == "AI_DECISION_DELAY":
        sim_scenario_name = "Stale LiDAR + delayed AI response"
    else:
        sim_scenario_name = "Stale LiDAR"

    ai_action_for_3d = "BYPASS_LEFT"
    if result.ai_decisions:
        rec_act = result.ai_decisions[0].recommended_action
        if rec_act == "SAFE_STOP_AND_HOLD":
            ai_action_for_3d = "SAFE_STOP"
        elif rec_act in ("BYPASS_LEFT", "BYPASS_RIGHT", "SAFE_STOP", "CONTINUE"):
            ai_action_for_3d = rec_act

    baseline_3d = run_simulation(scenario=sim_scenario_name, guarded=False)
    guarded_3d = run_simulation(scenario=sim_scenario_name, guarded=True, ai_action=ai_action_for_3d)

    scene_mode = st.radio(
        "3D Scene Mode:",
        [
            "Corridor Obstacle 3D Replay (Baseline vs. FaultShield)",
            "Multi-Rack Warehouse 3D Digital Twin (Variant B vs. Variant D)",
        ],
        index=0,
        horizontal=True,
    )

    world = WarehouseMap()
    with st.spinner("Preparing interactive 3D replay..."):
        if scene_mode.startswith("Corridor"):
            fig3d = build_3d_replay(baseline_3d, guarded_3d)
        else:
            fig3d = build_3d_warehouse_replay(
                world=world,
                baseline_traces=baseline_wh_result.trace,
                guarded_traces=guarded_wh_result.trace,
                goal_pos=custom_scenario.goal_pos,
            )
    st.plotly_chart(fig3d, use_container_width=True, key="faultshield_3d_replay")

    # ---------------- 2D TRAJECTORY DIAGNOSTICS ----------------
    st.markdown("---")
    st.subheader("📐 2D trajectory diagnostics")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Baseline status", baseline_3d.status)
    d2.metric("FaultShield status", guarded_3d.status)
    d3.metric("Baseline collision", "Yes" if baseline_3d.collision else "No")
    d4.metric("FaultShield collision", "Yes" if guarded_3d.collision else "No")

    # Replay Step Scrubber
    step_idx = (len(result.trace) - 1) if result.trace else 0
    if result.trace and len(result.trace) > 1:
        st.markdown("##### ⏱️ Live Simulation Step Scrubber & LiDAR Ray Sweeper")
        step_idx = st.slider(
            "Scrub timeline to inspect robot pose & active LiDAR beams at timestamp t:",
            0,
            len(result.trace) - 1,
            len(result.trace) - 1,
        )
        st_data = result.trace[step_idx]
        sc1, sc2, sc3, sc4, sc5 = st.columns(5)
        sc1.metric("Timestamp", f"{st_data.t:.2f} s")
        sc2.metric("GT Pose", f"({st_data.x_gt:.2f}, {st_data.y_gt:.2f})")
        sc3.metric("Odom Pose", f"({st_data.x_odom:.2f}, {st_data.y_odom:.2f})")
        sc4.metric("Governor Status", st_data.governor_status)
        sc5.metric("Active AI Action", st_data.ai_action or "None")

    # Visualizations
    col_map, col_plot = st.columns([1.1, 0.9])

    with col_map:
        fig_map = create_warehouse_figure(
            world=world,
            traces=result.trace,
            goal_pos=custom_scenario.goal_pos,
            title=f"Warehouse Floorplan ({m.variant_id} - {custom_scenario.name})",
            show_lidar_step=step_idx if result.trace else None,
        )
        st.plotly_chart(fig_map, use_container_width=True)

    with col_plot:
        fig_time = create_telemetry_timeline_figure(result.trace)
        st.plotly_chart(fig_time, use_container_width=True)

    st.subheader("📋 Event timeline (Baseline vs. FaultShield)")
    event_col1, event_col2 = st.columns(2)
    with event_col1:
        st.markdown("**Baseline controller**")
        for event in baseline_3d.events:
            st.write(f"• {event}")
    with event_col2:
        st.markdown("**FaultShield controller**")
        for event in guarded_3d.events:
            st.write(f"• {event}")

    if guarded_3d.rejected_ai_action:
        st.warning(f"Safety governor rejected action: {guarded_3d.rejected_ai_action}")


# ---------------- TAB 2: 4-VARIANT BENCHMARK ----------------
with tab2:
    st.markdown("### ⚖️ Side-by-Side 4-Variant Benchmark")
    st.markdown(
        "Evaluating the proposed research gap: comparing **A** (Healthy Baseline), **B** (Unmitigated Faults), "
        "**C** (Conventional Rule-Based Fallback), and **D** (FaultShield with Nemotron & Safety Governor)."
    )

    if run_benchmark_button or "comparison_results" not in st.session_state:
        with st.spinner("Executing all 4 system variants under identical scenario conditions..."):
            variants = ["Variant_A", "Variant_B", "Variant_C", "Variant_D"]
            comp_results = {}
            comp_metrics = []
            for v in variants:
                res = runner.run_episode(custom_scenario, variant_id=v, seed=seed)
                comp_results[v] = res
                comp_metrics.append(res.metrics)
            st.session_state.comparison_results = comp_results
            st.session_state.comparison_metrics = comp_metrics

    comp_metrics = st.session_state.comparison_metrics
    summaries = [compute_benchmark_summary([m]) for m in comp_metrics]

    # Grouped Bar Chart
    fig_comp = create_variant_comparison_chart(summaries)
    st.plotly_chart(fig_comp, use_container_width=True)

    # Comparative Metrics Table
    summary_data = []
    for m in comp_metrics:
        summary_data.append({
            "System Variant": m.variant_id,
            "Collision?": "💥 YES" if m.collision_occurred else "✅ NO",
            "Goal Reached?": "🏆 YES" if m.goal_reached else "⏸ NO",
            "Path Length (m)": m.path_length_meters,
            "Min Clearance (m)": m.min_obstacle_clearance_meters,
            "Stale Decisions Rejected": m.stale_decisions_rejected,
            "Stale Decisions Executed": m.stale_decisions_accepted,  # Must be 0
            "Duration (s)": m.duration_sec,
        })
    df_comp = pd.DataFrame(summary_data)
    st.dataframe(df_comp, use_container_width=True, hide_index=True)

    st.markdown("""
    #### 💡 Research Finding & Analysis:
    - **Variant B (Unmitigated)** blindly follows the corrupted telemetry and crashes into obstacles.
    - **Variant C (Conventional Rule-based Fallback)** halts immediately, preventing the crash, but permanently abandons the mission without context-aware recovery.
    - **Variant D (FaultShield)** detects the anomaly, queries NVIDIA Nemotron for an allowlisted recovery action, validates the freshness via the Safety Governor, and attempts a safe maneuver.
    - When artificial latency exceeds **0.50s**, the Safety Governor **rejects the stale Nemotron decision** and executes an immediate local safe stop, preventing outdated physical actions.
    """)


# ---------------- TAB 3: TELEMETRY & DECISION LOG ----------------
with tab3:
    st.markdown("### 📡 Sensor Integrity & AI Decision Chain")
    res: EpisodeRunResult = st.session_state.current_result

    col_inc, col_ai = st.columns(2)

    with col_inc:
        st.subheader("Sensor-Integrity Monitor (SIM) Incidents")
        if not res.incidents:
            st.info("No sensor integrity anomalies triggered.")
        else:
            for inc in res.incidents:
                with st.expander(f"⚠️ {inc.incident_id} at t={inc.timestamp:.2f}s | {inc.primary_fault}"):
                    st.write(f"**Risk Level:** `{inc.risk_level}`")
                    st.write(f"**Diagnostic Details:** {inc.raw_discrepancy_details}")
                    st.json(inc.to_dict())

    with col_ai:
        st.subheader("NVIDIA Nemotron Recovery Decisions")
        if not res.ai_decisions:
            st.info("No AI recovery requests dispatched.")
        else:
            for dec in res.ai_decisions:
                with st.expander(f"🤖 {dec.decision_id} (Latency: {dec.latency_sec:.3f}s)"):
                    st.write(f"**Recommended Action:** `{dec.recommended_action}`")
                    st.write(f"**Confidence:** {dec.confidence:.2f}")
                    st.write(f"**Root Cause Diagnosis:** {dec.root_cause_diagnosis}")
                    st.write(f"**Safety Rationale:** {dec.safety_rationale}")
                    st.write(f"**Simulated / Live:** {'Mock Engine' if dec.is_simulated else 'Live Nebius Token Factory'}")
                    st.code(dec.raw_response, language="json")

    st.markdown("---")
    st.subheader("Independent Safety Governor Execution Log")
    if res.governor_log:
        gov_rows = []
        for i, g in enumerate(res.governor_log):
            if g.status != "NORMAL_PASS":
                gov_rows.append({
                    "Step": i,
                    "Status": g.status,
                    "Executed V": g.executed_v,
                    "Executed ω": g.executed_omega,
                    "Stale Age (s)": round(g.stale_age_sec, 3),
                    "Governor Rationale": g.reason,
                })
        if gov_rows:
            st.dataframe(pd.DataFrame(gov_rows), use_container_width=True)
        else:
            st.write("All commands passed standard envelope.")


# ---------------- TAB 4: BATCH STATISTICAL EVALUATION ----------------
with tab4:
    st.markdown("### 📊 Automated Multi-Trial Statistical Evaluation")
    st.markdown("Run repeated trials with randomized seeds to evaluate statistical variance across scenarios.")

    num_trials = st.slider("Number of Monte-Carlo Trials per Variant", min_value=3, max_value=20, value=5)

    if st.button("🚀 Execute Batch Benchmark Run"):
        progress_bar = st.progress(0.0)
        batch_records = []
        variants = ["Variant_A", "Variant_B", "Variant_C", "Variant_D"]
        total_runs = num_trials * len(variants)
        run_count = 0

        for s_idx in range(num_trials):
            trial_seed = 100 + s_idx * 17
            for v in variants:
                res = runner.run_episode(custom_scenario, variant_id=v, seed=trial_seed)
                batch_records.append(res.metrics)
                run_count += 1
                progress_bar.progress(run_count / total_runs)

        progress_bar.empty()
        st.success(f"Completed {total_runs} simulation episodes across 4 variants!")

        # Summaries
        batch_summary_data = []
        for v in variants:
            v_metrics = [m for m in batch_records if m.variant_id == v]
            s = compute_benchmark_summary(v_metrics)
            if s:
                batch_summary_data.append({
                    "Variant": s.variant_id,
                    "Trials": s.num_episodes,
                    "Collision Rate (%)": s.collision_rate * 100,
                    "Recovery Rate (%)": s.recovery_rate * 100,
                    "Goal Completion (%)": s.goal_completion_rate * 100,
                    "Mean Duration (s)": s.mean_duration_sec,
                    "Mean Path Length (m)": s.mean_path_length_m,
                    "Stale Acceptance (%)": s.stale_acceptance_rate * 100,
                })

        st.dataframe(pd.DataFrame(batch_summary_data), use_container_width=True, hide_index=True)

        # Download CSV
        raw_df = pd.DataFrame([m.to_dict() for m in batch_records])
        csv_data = raw_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Benchmark Results (CSV)",
            data=csv_data,
            file_name=f"faultshield_benchmark_{custom_scenario.id}.csv",
            mime="text/csv",
        )


# ---------------- TAB 5: RESEARCH GAP & ARCHITECTURE ----------------
with tab5:
    st.markdown("### 🔬 Research Gap & Scientific Foundation")
    st.markdown("""
    #### 1. Why Existing Solutions Are Not Enough
    - **Manipulation failure diagnosis (VLA-FixBench, ICML 2026):** Established benchmarks evaluate VLMs diagnosing manipulation failures, but do not address mobile robots under corrupted telemetry.
    - **Recovery benchmarks (LIBERO-Recover, 2026):** Evaluates retries and adaptation in high-level tasks, but assumes accurate state estimation.
    - **Latency-Aware Architectures (LARA, 2026):** Explores asynchronous inference delays, but does not address the interaction between sensor corruptions and stale LLM decisions.
    - **Patent Landscape (NVIDIA US20260091502A1, Sony US20250199539A1, ABB CN116368444B):** Generic failure detection and uncertainty-aware control are existing claims. FaultShield distinguishes itself by evaluating cross-layer interactions.

    #### 2. System Architecture Flow
    ```
    [ 2D Warehouse Simulation ] (Ground Truth: x, y, theta)
                 │
                 ▼
    [ Sensor Suite ] (LiDAR, Odometry, IMU)
                 │
                 ▼
    [ Fault Injection Engine ] (Freeze, Dropout, Drift, Bias)
                 │
                 ▼
    [ Sensor-Integrity Monitor ] (Freshness, Variance, Plausibility)
                 │  (Flags Anomaly / Incident)
                 ▼
    [ NVIDIA Nemotron Agent ] (via Nebius Token Factory)
                 │  (Recommends Allowlisted Action)
                 ▼
    [ Independent Safety Governor ] (TTL Check < 0.5s, Envelope Clamp)
                 │  (ACCEPTED / REJECTED_STALE)
                 ▼
    [ Robot Actuation & Recovery ] (Deterministic Execution)
    ```

    #### 3. Nebius Token Factory Integration & 3D Digital Twin
    - Target Model: `nvidia/nemotron-3-super-120b-a12b`
    - Endpoint: `https://api.tokenfactory.nebius.com/v1/`
    - Free Hackathon Promotion Code: `NEBIUS-DEVPOST-GLOBAL26` ($25 credit)
    - Fallback: High-fidelity deterministic local engine for zero-network testing.
    - **3D Visualization Transparency:** FaultShield 3D (Version 2) renders the 2D kinematic simulation in an interactive 3D digital-twin view with 3D robot chassis/wheels, obstacles, floor grid, and Play/Pause/timeline controls. It is not yet a full 3D rigid-body physics simulator (such as PyBullet or Webots).
    """)
