"""
Command-line Benchmark Runner for FaultShield.
Executes reproducible evaluations across all 5 benchmark scenarios and 4 system variants,
outputting statistical tables and CSV exports for hackathon submission evidence.
"""

import argparse
import json
import os
import sys

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
from faultshield.faults.scenarios import BENCHMARK_SCENARIOS
from faultshield.agent.nemotron import NemotronRecoveryAgent
from faultshield.benchmark.runner import BenchmarkRunner
from faultshield.benchmark.metrics import compute_benchmark_summary


def run_full_benchmark(num_trials: int = 5, output_dir: str = "results"):
    os.makedirs(output_dir, exist_ok=True)
    agent = NemotronRecoveryAgent(use_mock=True)
    runner = BenchmarkRunner(nemotron_agent=agent)

    variants = ["Variant_A", "Variant_B", "Variant_C", "Variant_D"]
    all_metrics = []
    summaries = []

    print("=" * 80)
    print(" [FAULTSHIELD] CROSS-LAYER RELIABILITY BENCHMARK SUITE")
    print(f" Running {len(BENCHMARK_SCENARIOS)} scenarios x {len(variants)} variants x {num_trials} trials")
    print("=" * 80)

    for sc_id, scenario in BENCHMARK_SCENARIOS.items():
        print(f"\n--- Scenario: {scenario.name} ---")
        for v in variants:
            v_metrics = []
            for trial in range(num_trials):
                seed = 1000 + trial * 41
                res = runner.run_episode(scenario, variant_id=v, seed=seed)
                v_metrics.append(res.metrics)
                all_metrics.append(res.metrics)

            summary = compute_benchmark_summary(v_metrics)
            if summary:
                summaries.append({
                    "Scenario": sc_id,
                    "Variant": v,
                    "Trials": summary.num_episodes,
                    "Collision Rate (%)": summary.collision_rate * 100,
                    "Recovery Rate (%)": summary.recovery_rate * 100,
                    "Goal Completion (%)": summary.goal_completion_rate * 100,
                    "Mean Duration (s)": summary.mean_duration_sec,
                    "Mean Path Length (m)": summary.mean_path_length_m,
                    "Stale Rejections Blocked": sum(m.stale_decisions_rejected for m in v_metrics),
                    "Stale Acceptance (%)": summary.stale_acceptance_rate * 100,
                })
                print(
                    f"  [{v}] Collision: {summary.collision_rate*100:5.1f}% | "
                    f"Recovery: {summary.recovery_rate*100:5.1f}% | "
                    f"Goal: {summary.goal_completion_rate*100:5.1f}% | "
                    f"Path: {summary.mean_path_length_m:4.1f}m"
                )

    # Save CSV
    summary_df = pd.DataFrame(summaries)
    summary_csv_path = os.path.join(output_dir, "benchmark_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)

    raw_df = pd.DataFrame([m.to_dict() for m in all_metrics])
    raw_csv_path = os.path.join(output_dir, "benchmark_raw_episodes.csv")
    raw_df.to_csv(raw_csv_path, index=False)

    print("\n" + "=" * 80)
    print(" BENCHMARK COMPLETED SUCCESSFULLY!")
    print(f" Summary saved to: {summary_csv_path}")
    print(f" Raw episodes saved to: {raw_csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FaultShield Benchmarks")
    parser.add_argument("--trials", type=int, default=5, help="Number of trials per scenario/variant")
    parser.add_argument("--out", type=str, default="results", help="Output directory")
    args = parser.parse_args()
    run_full_benchmark(num_trials=args.trials, output_dir=args.out)
