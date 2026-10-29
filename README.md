# 🛡️ FaultShield: Cross-Layer Reliability Benchmark & Recovery Agent

> **A Cross-Layer Reliability Benchmark and Recovery Agent for Simulated Mobile Robots**  
> *Physical AI Track — NVIDIA Nemotron — Nebius Token Factory — Zero Hardware Required*

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://python.org)
[![Nebius Token Factory](https://img.shields.io/badge/Nebius-Token%20Factory-3b82f6.svg)](https://tokenfactory.nebius.ai/)
[![NVIDIA Nemotron](https://img.shields.io/badge/NVIDIA-Nemotron--4--340B-76b900.svg)](https://build.nvidia.com)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Interactive%20UI-ff4b4b.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 The Research Opportunity: Why FaultShield?

Recent literature establishes that robot failure diagnosis (**VLA-FixBench**, ICML 2026), recovery benchmarks (**LIBERO-Recover**, 2026), and latency-aware controllers (**LARA**, 2026) are active domains. Furthermore, existing patents (NVIDIA US20260091502A1, Sony US20250199539A1, ABB CN116368444B) address high-level failure explanations and generic uncertainty.

### The Remaining Gap
Rather than claiming a generic AI failure detector, **FaultShield addresses the unresolved cross-layer interaction:**
> *When an embodied AI agent receives delayed, corrupted, or contradictory sensor data—and its own cloud reasoning pipeline is subject to latency or packet jitter—can it detect when **not** to act, enter a safe state, and recover reliably?*

FaultShield solves this through a controlled, repeatable evaluation suite comparing a differential-drive robot across **four distinct system architectures**.

---

## 🏗️ System Architecture

```
                       ┌──────────────────────────────────────────────┐
                       │   2D Warehouse World (Ground Truth Scoring)  │
                       └──────────────────────┬───────────────────────┘
                                              │ (x, y, θ)
                                              ▼
                       ┌──────────────────────────────────────────────┐
                       │  Sensor Suite (2D LiDAR, Odometry, IMU Gyro) │
                       └──────────────────────┬───────────────────────┘
                                              │ Raw Telemetry
                                              ▼
                       ┌──────────────────────────────────────────────┐
                       │     Configurable Fault Injection Engine      │
                       │ (Freeze, Range Dropout, Odom Drift, IMU Bias)│
                       └──────────────────────┬───────────────────────┘
                                              │ Corrupted Telemetry Stream
                                              ▼
                       ┌──────────────────────────────────────────────┐
                       │        Sensor-Integrity Monitor (SIM)        │
                       │ (Freshness check, Variance, Cross-modal test)│
                       └──────────────────────┬───────────────────────┘
                                              │ Incident Report (JSON)
                                              ▼
                       ┌──────────────────────────────────────────────┐
                       │     NVIDIA Nemotron Recovery Agent           │
                       │      (via Nebius Token Factory API)          │
                       │  - Recommends strictly allowlisted action    │
                       └──────────────────────┬───────────────────────┘
                                              │ Action Dispatch (timestamped)
                                              ▼
                       ┌──────────────────────────────────────────────┐
                       │         Independent Safety Governor          │
                       │  - TTL Check (< 0.50s; rejects stale AI recs)│
                       │  - Kinematic Allowlist & Speed Clamping      │
                       │  - Critical obstacle proximity override      │
                       └──────────────────────┬───────────────────────┘
                                              │ Validated Velocity (v, ω)
                                              ▼
                       ┌──────────────────────────────────────────────┐
                       │   Robot Kinematic Actuation & Safe Recovery  │
                       └──────────────────────────────────────────────┘
```

---

## 🧪 The Five Controlled Benchmark Experiments

| Experiment | Title | Description | Primary Failure Mode Tested |
|---|---|---|---|
| **Exp 1** | **Healthy Baseline** | Robot receives valid measurements and navigates across the warehouse. | Establishes zero-fault travel time and navigation envelope. |
| **Exp 2** | **Misleading Range Sensor** | LiDAR driver freezes on clear corridor readings as robot approaches storage racks. | Blind collision due to stale sensor buffer. |
| **Exp 3** | **Contradictory Measurements** | IMU gyro develops acute angular drift while wheel encoders report straight path. | Cross-sensor kinematic divergence & state disorientation. |
| **Exp 4** | **Delayed AI Decision** | Artificial 2.2s latency injected into remote reasoning service. | Tests Safety Governor's ability to reject stale recommendations. |
| **Exp 5** | **Compound Cross-Layer Faults** | Simultaneous frozen LiDAR, wheel slip drift, and inference latency. | Tests resilience against compound simultaneous failures. |

---

## ⚖️ Four-Variant Comparative Evaluation Model

FaultShield tests every scenario across 4 variants to rigorously measure whether AI recovery adds genuine value:

- **Variant A (Healthy Baseline):** Standard controller, no faults injected.
- **Variant B (Unmitigated Controller):** Standard controller receiving corrupted telemetry (demonstrates failure/crashes).
- **Variant C (Conventional Monitor + Fixed Fallback):** Rule-based monitor stops the robot unconditionally upon anomaly detection (non-LLM baseline).
- **Variant D (FaultShield System):** Sensor-Integrity Monitor + NVIDIA Nemotron Recovery Agent + Independent Safety Governor.

---

## 🚀 Quickstart Guide

### 1. Environment Setup
The project comes pre-configured with a virtual environment:
```powershell
# Create venv if needed
python -m venv .venv

# Activate venv in PowerShell
.\.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 2. Configure Nebius Token Factory (Optional / Recommended)
Copy `.env.example` to `.env` and set your Token Factory credentials:
```bash
cp .env.example .env
```
In `.env`:
```env
NEBIUS_API_KEY="your_nebius_api_key_here"
NEBIUS_BASE_URL="https://api.tokenfactory.nebius.ai/v1"
NEMOTRON_MODEL="nvidia/nemotron-4-340b-instruct"
```
> **Note:** If no API key is provided or you do not have network access, FaultShield seamlessly activates its high-fidelity deterministic local engine, enabling complete offline testing and demonstration!

### 3. Run Automated Tests
```powershell
.\.venv\Scripts\pytest.exe -v
```

### 4. Launch the Interactive Dashboard
```powershell
.\.venv\Scripts\streamlit.exe run app.py
```
Open your browser at `http://localhost:8501`.

### 5. Run the Headless Batch Benchmark CLI
```powershell
.\.venv\Scripts\python.exe run_benchmark.py --trials 5 --out results
```
Generates complete statistical CSV reports in `results/benchmark_summary.csv` and `results/benchmark_raw_episodes.csv`.

---

## 📊 Key Track Metrics Tracked

- **Collision Rate (%):** Primary physical safety metric.
- **Recovery Success Rate (%):** Safe task resumption without incident.
- **Stale-Response Acceptance Rate:** Percentage of expired AI recommendations executed (**Must be 0.0%**).
- **Time-to-Safe-State (s):** Latency between fault occurrence and safe transition.
- **Fault Detection Latency (s):** Time taken by SIM to detect anomaly.
- **Decision Age & Inference Latency (s):** Tracked against the 0.5s TTL window.

---

## 📜 License

MIT License. Designed and built for the **Nebius Global AI Hackathon (Physical AI Track)**.
