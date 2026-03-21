"""
calibrate_regime_thresholds.py
===============================
Methodological Appendix -- Empirical Calibration of Regime Detection Thresholds

The function detect_regime() in Baseline.py classifies each time step into one
of three regimes (unclassified / STR / RSR) using four thresholds defined in
SimulationConfig:

    str_drift_max  -- maximum mean absolute increment for STR classification
    str_delta_min  -- minimum spatio-temporal std for STR classification
    str_delta_max  -- maximum spatio-temporal std for STR classification
    rsr_delta_min  -- minimum spatio-temporal std for RSR classification

This script observes the empirical distributions of the two quantities
computed inside detect_regime() -- recent_delta and recent_drift -- and
compares them against the current thresholds. It provides the empirical
basis for the threshold values used throughout the paper.

Two important conventions are matched exactly to detect_regime():

    recent_drift:
        Computed from raw increments tau[t] - tau[t-1], NOT divided by dt.
        Dividing by dt would inflate drift by 1/dt = 12.5 and produce
        thresholds inconsistent with the detector.

    recent_delta:
        np.std(tau_history[t-20:t]) computes std over a spatio-temporal
        window of shape (20, N) -- not the spatial std over N nodes at a
        single instant (which is what compute_delta() returns).
        These are two distinct quantities. This script targets the
        spatio-temporal version used inside detect_regime().

Requires: Baseline.py in the same directory.

Usage:
    python calibrate_regime_thresholds.py

Output:
    Printed distributions and sampled values -- no figures generated.
"""

import numpy as np
from Baseline import SimulationConfig, run_simulation

# ============================================================
# Run baseline simulation
# ============================================================

result      = run_simulation(SimulationConfig(seed=42))
tau_history = result["tau_history"]
cfg         = SimulationConfig(seed=42)

# ============================================================
# Collect full distributions (t=21 to T-1)
# ============================================================

all_delta = []
all_drift = []

for t in range(21, cfg.T):
    recent_delta = float(np.std(tau_history[t - 20:t]))
    dtau         = tau_history[t] - tau_history[t - 1]   # raw, no /dt
    recent_drift = float(np.mean(np.abs(dtau)))
    all_delta.append(recent_delta)
    all_drift.append(recent_drift)

all_delta = np.array(all_delta)
all_drift = np.array(all_drift)

# ============================================================
# Distribution summary
# ============================================================

print("=== Current STR/RSR thresholds in SimulationConfig ===")
print(f"  str_drift_max : {cfg.str_drift_max}")
print(f"  str_delta_min : {cfg.str_delta_min}")
print(f"  str_delta_max : {cfg.str_delta_max}")
print(f"  rsr_delta_min : {cfg.rsr_delta_min}")

print("\n=== Distribution of recent_delta (spatio-temporal window, 20 x N) ===")
print(f"  min    : {all_delta.min():.6f}")
print(f"  max    : {all_delta.max():.6f}")
print(f"  mean   : {all_delta.mean():.6f}")
print(f"  median : {np.median(all_delta):.6f}")
print(f"  steps > str_delta_max ({cfg.str_delta_max}) : {(all_delta > cfg.str_delta_max).sum()}")
print(f"  steps < str_delta_min ({cfg.str_delta_min}) : {(all_delta < cfg.str_delta_min).sum()}")

print("\n=== Distribution of recent_drift (raw increment, no /dt) ===")
print(f"  min    : {all_drift.min():.6f}")
print(f"  max    : {all_drift.max():.6f}")
print(f"  mean   : {all_drift.mean():.6f}")
print(f"  median : {np.median(all_drift):.6f}")
print(f"  steps > str_drift_max ({cfg.str_drift_max}) : {(all_drift > cfg.str_drift_max).sum()}")

# ============================================================
# Sampled values every 30 steps
# ============================================================

print("\n=== Sampled values (every 30 steps) ===")
print(f"{'t':>5}  {'recent_delta':>14}  {'recent_drift_raw':>17}  {'STR':>6}")
for t in range(31, cfg.T):
    if t % 30 == 0:
        recent_delta = float(np.std(tau_history[t - 20:t]))
        dtau         = tau_history[t] - tau_history[t - 1]
        recent_drift = float(np.mean(np.abs(dtau)))
        str_ok = (
            recent_drift < cfg.str_drift_max
            and cfg.str_delta_min < recent_delta < cfg.str_delta_max
        )
        print(f"t={t:>3}  recent_delta={recent_delta:.6f}"
              f"  recent_drift_raw={recent_drift:.6f}  STR={str_ok}")
print(f"  median : {np.median(all_drift):.6f}")
print(f"  steps > str_drift_max ({cfg.str_drift_max}) : {(all_drift > cfg.str_drift_max).sum()}")

# --------------------------------------------------------
# Sampled values every 30 steps
# --------------------------------------------------------
print("\n=== Sampled values (every 30 steps) ===")
print(f"{'t':>5}  {'recent_delta':>14}  {'recent_drift_raw':>17}  {'STR':>6}")
for t in range(31, cfg.T):
    if t % 30 == 0:
        recent_delta = float(np.std(tau_history[t - 20:t]))
        dtau         = tau_history[t] - tau_history[t - 1]
        recent_drift = float(np.mean(np.abs(dtau)))
        str_ok = (
            recent_drift < cfg.str_drift_max
            and cfg.str_delta_min < recent_delta < cfg.str_delta_max
        )
        print(f"t={t:>3}  recent_delta={recent_delta:.6f}"
              f"  recent_drift_raw={recent_drift:.6f}  STR={str_ok}")
