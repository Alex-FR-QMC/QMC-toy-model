"""
mcq_regulated.py
================
Section 5 — MCQ-Inspired Minimal Regulated Dynamics

Introduces four operative mechanisms into the baseline tensional field,
each approximating a construct from Chapter 1 of the QMC paradigm:

    anti-synchronisation  ->  approx. MI  (maintains integrable local gap)
    threshold detection   ->  approx. RR3 (detects drift toward boundary of V)
    directed perturbation ->  approx. MV  (morphodynamic reconfiguration)
    short local memory    ->  approx. C_T (comparative memory over window)

These mechanisms are intentionally minimal. They do not implement the full
architecture of Chapter 2. Their purpose is to demonstrate that a small
set of MCQ-inspired constraints is sufficient to restore and maintain
viability where the unregulated baseline fails.

Requires: Baseline.py in the same directory.

Usage:
    python mcq_regulated.py

Outputs:
    figures/comparison.png   -- baseline vs regulated superposed
    printed comparison table -- key metrics for both systems
"""

import os
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

from Baseline import (
    SimulationConfig, make_graph, adjacency_matrix,
    compute_delta, compute_g_on_edges, in_viability,
    detect_regime, first_permanent_exit, first_time_below,
    returned_to_viability, run_simulation
)


# ============================================================
# MCQ-inspired regulated dynamics config
# ============================================================

@dataclass
class RegulatedConfig:
    base: SimulationConfig = field(default_factory=SimulationConfig)

    # --- Anti-synchronisation ---
    # Opposes local homogenisation: pushes tau_i away from neighbours
    # when local contrast |tau_i - tau_j| falls below anti_sync_threshold
    anti_sync_strength: float  = 0.18
    anti_sync_threshold: float = 0.04   # below this, anti-sync activates

    # --- Threshold detection (approx. RR3) ---
    # When Delta(t) < delta_alert, directed perturbation triggers
    delta_alert: float = 0.06

    # --- Directed perturbation (approx. MV) ---
    # Targets the most homogeneous nodes and injects structured noise
    directed_strength: float  = 0.12
    directed_fraction: float  = 0.33   # fraction of nodes targeted

    # --- Short local memory (approx. C_T) ---
    # Each node is pushed away from its own recent mean
    memory_window: int   = 8
    memory_strength: float = 0.08


# ============================================================
# Regulated simulation
# ============================================================

def run_regulated(rcfg: RegulatedConfig) -> Dict[str, Any]:
    cfg = rcfg.base
    rng = np.random.default_rng(cfg.seed)

    G = make_graph(cfg)
    A = adjacency_matrix(G, cfg.N)

    # Initialise field
    tau = rng.uniform(cfg.tau_min_init, cfg.tau_max_init, cfg.N)

    # History
    tau_history    = np.zeros((cfg.T, cfg.N))
    delta_history  = np.zeros(cfg.T)
    g_history      = np.zeros(cfg.T)
    viable_history = np.zeros(cfg.T, dtype=bool)
    regime_history = np.zeros(cfg.T, dtype=int)

    # Short memory buffer  [T x N]
    mem_buffer = np.full((rcfg.memory_window, cfg.N), np.mean(tau))

    dtau = np.zeros(cfg.N)

    for t in range(cfg.T):
        delta     = compute_delta(tau)
        g         = compute_g_on_edges(tau, A)
        is_viable = in_viability(delta, g, cfg)

        delta_history[t]  = delta
        g_history[t]      = g
        viable_history[t] = is_viable
        regime_history[t] = detect_regime(t, tau_history, dtau, cfg)

        # --- Baseline dynamics ---
        gamma_pp  = rng.normal(0.0, cfg.noise_std, cfg.N)
        degree    = np.sum(A, axis=1)
        coupling  = cfg.coupling_strength * (A @ tau - degree * tau)
        reflexive = -cfg.reflexive_strength * (tau - np.mean(tau))

        # --- MCQ mechanism 1 : anti-synchronisation ---
        anti_sync = np.zeros(cfg.N)
        for i in range(cfg.N):
            for j in range(cfg.N):
                if A[i, j] > 0:
                    diff = tau[i] - tau[j]
                    if abs(diff) < rcfg.anti_sync_threshold:
                        # Push away from neighbour
                        sign = 1.0 if diff >= 0 else -1.0
                        anti_sync[i] += rcfg.anti_sync_strength * sign * (
                            rcfg.anti_sync_threshold - abs(diff)
                        )

        # --- MCQ mechanism 2 : threshold detection + directed perturbation ---
        directed = np.zeros(cfg.N)
        if delta < rcfg.delta_alert:
            # Identify most homogeneous nodes (smallest local contrast)
            local_contrast = np.array([
                np.mean([abs(tau[i] - tau[j]) for j in range(cfg.N) if A[i, j] > 0])
                if np.sum(A[i]) > 0 else 1.0
                for i in range(cfg.N)
            ])
            n_target  = max(1, int(cfg.N * rcfg.directed_fraction))
            targets   = np.argsort(local_contrast)[:n_target]
            # Inject structured noise pushing toward corridor centre
            for idx in targets:
                direction = 1.0 if tau[idx] < np.mean(tau) else -1.0
                directed[idx] = rcfg.directed_strength * direction * rng.uniform(0.5, 1.5)

        # --- MCQ mechanism 3 : short local memory ---
        mem_buffer = np.roll(mem_buffer, 1, axis=0)
        mem_buffer[0] = tau.copy()
        local_mean_history = np.mean(mem_buffer, axis=0)
        memory_push = rcfg.memory_strength * (tau - local_mean_history)

        # --- Full update ---
        dtau = cfg.dt * (
            coupling
            + reflexive
            + 0.48 * gamma_pp
            + anti_sync
            + directed
            - memory_push
        )
        tau = np.clip(tau + dtau, cfg.tau_clip_min, cfg.tau_clip_max)
        tau_history[t] = tau.copy()

    return {
        "config":         rcfg,
        "tau_history":    tau_history,
        "delta_history":  delta_history,
        "g_history":      g_history,
        "viable_history": viable_history,
        "regime_history": regime_history,
        "t_exit_V":          first_permanent_exit(viable_history),
        "t_delta_collapse":  first_time_below(delta_history, cfg.delta_floor),
        "t_g_collapse":      first_time_below(g_history, cfg.g_floor),
        "returned_to_V":     returned_to_viability(viable_history),
        "final_delta":       float(delta_history[-1]),
        "final_g":           float(g_history[-1]),
        "fraction_viable":   float(np.mean(viable_history)),
    }


# ============================================================
# Comparison plot : baseline vs regulated (superposed)
# ============================================================

def plot_comparison(
    baseline_result: Dict[str, Any],
    regulated_result: Dict[str, Any],
    save_path: Optional[str] = None
) -> None:
    cfg  = baseline_result["config"]
    rcfg = regulated_result["config"]
    T    = cfg.T
    ts   = np.arange(T)

    fig = plt.figure(figsize=(14, 13))
    fig.suptitle("QMC Toy Model — Baseline vs MCQ-Regulated", fontsize=15, y=0.98)
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.32)

    ax_field  = fig.add_subplot(gs[0, 0])
    ax_delta  = fig.add_subplot(gs[0, 1])
    ax_g      = fig.add_subplot(gs[1, 0])
    ax_viable = fig.add_subplot(gs[1, 1])
    ax_regime = fig.add_subplot(gs[2, :])

    C_BASE = "#aaaaaa"
    C_REG  = "#2166ac"

    # --- Field trajectories ---
    for i in range(cfg.N):
        ax_field.plot(baseline_result["tau_history"][:, i],
                      lw=0.8, color=C_BASE, alpha=0.5)
    for i in range(cfg.N):
        ax_field.plot(regulated_result["tau_history"][:, i],
                      lw=1.2, color=C_REG, alpha=0.7)
    ax_field.plot([], [], lw=1.5, color=C_BASE, label="Baseline")
    ax_field.plot([], [], lw=1.5, color=C_REG,  label="Regulated")
    ax_field.set_title("Tensional Field tau_i(t)")
    ax_field.set_xlabel("Time steps")
    ax_field.set_ylabel("tau_i")
    ax_field.legend(fontsize=9)

    # --- Delta ---
    ax_delta.plot(ts, baseline_result["delta_history"],
                  lw=1.5, color=C_BASE, label="Baseline")
    ax_delta.plot(ts, regulated_result["delta_history"],
                  lw=2.0, color=C_REG,  label="Regulated")
    ax_delta.axhline(cfg.delta_crit,  ls="--", lw=1.5, color="red",    label="delta_crit")
    ax_delta.axhline(rcfg.delta_alert if hasattr(rcfg, 'delta_alert') else 0.06,
                     ls=":",  lw=1.2, color="orange", label="delta_alert")
    ax_delta.axhline(cfg.delta_floor, ls=":",  lw=1.0, color="gray",   label="delta_floor")
    ax_delta.set_title("Morphodynamic Corridor Delta(t)")
    ax_delta.set_xlabel("Time steps")
    ax_delta.set_ylabel("Delta")
    ax_delta.legend(fontsize=8)

    # --- G ---
    ax_g.plot(ts, baseline_result["g_history"],
              lw=1.5, color=C_BASE, label="Baseline")
    ax_g.plot(ts, regulated_result["g_history"],
              lw=2.0, color=C_REG,  label="Regulated")
    ax_g.axhline(cfg.g_min,   ls="--", lw=1.5, color="red",  label="G_min")
    ax_g.axhline(cfg.g_floor, ls=":",  lw=1.0, color="gray", label="G_floor")
    ax_g.set_title("Transformable Gradient G(t)")
    ax_g.set_xlabel("Time steps")
    ax_g.set_ylabel("G")
    ax_g.legend(fontsize=8)

    # --- Viability ---
    ax_viable.plot(ts, baseline_result["viable_history"].astype(int),
                   lw=1.5, color=C_BASE, label="Baseline", alpha=0.8)
    ax_viable.plot(ts, regulated_result["viable_history"].astype(int),
                   lw=2.0, color=C_REG,  label="Regulated")
    ax_viable.set_ylim(-0.1, 1.3)
    ax_viable.set_yticks([0, 1])
    ax_viable.set_yticklabels(["No", "Yes"])
    ax_viable.set_title("Membership in Viability Domain V")
    ax_viable.set_xlabel("Time steps")
    ax_viable.legend(fontsize=9)

    # --- Regimes ---
    rh_b = baseline_result["regime_history"]
    rh_r = regulated_result["regime_history"]

    offset = 0.15
    for rh, color, label, y_off in [
        (rh_b, C_BASE, "Baseline",   offset),
        (rh_r, C_REG,  "Regulated", -offset),
    ]:
        str_t = ts[rh == 1]
        rsr_t = ts[rh == 2]
        if len(str_t) > 0:
            ax_regime.scatter(str_t, np.zeros(len(str_t)) + y_off,
                              s=5, color=color, alpha=0.6,
                              label=f"STR {label}" if y_off > 0 else None)
        if len(rsr_t) > 0:
            ax_regime.scatter(rsr_t, np.ones(len(rsr_t)) + y_off,
                              s=5, color=color, alpha=0.9,
                              label=f"RSR {label}" if y_off > 0 else None,
                              marker="D")

    ax_regime.set_xlim(0, T)
    ax_regime.set_ylim(-0.5, 1.8)
    ax_regime.set_yticks([0, 1])
    ax_regime.set_yticklabels(["STR", "RSR"])
    ax_regime.set_title("Dynamic Regimes — Baseline (grey) vs Regulated (blue)")
    ax_regime.set_xlabel("Time steps")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)

    # --------------------------------------------------------
    # Section 5 — MCQ-inspired regulated dynamics
    # Calibrated parameters — reproduce figures/comparison.png
    #
    # MCQ mechanism mapping:
    #   anti_sync       → approx. MI  (maintains integrable gap)
    #   delta_alert     → approx. RR3 (drift detection toward dV)
    #   directed        → approx. MV  (morphodynamic reconfiguration)
    #   memory_push     → approx. C_T (short comparative memory)
    # --------------------------------------------------------

    baseline = run_simulation(SimulationConfig(seed=42))

    # Calibrated regulated config — matches results reported in Section 5
    rcfg = RegulatedConfig(
        base                = SimulationConfig(seed=42),
        anti_sync_strength  = 0.35,    # MI: force maintaining local contrast
        anti_sync_threshold = 0.12,    # MI: activation threshold on |tau_i - tau_j|
        delta_alert         = 0.07,    # RR3: drift detection threshold
        directed_strength   = 0.28,    # MV: intensity of directed reconfiguration
        directed_fraction   = 0.40,    # MV: fraction of nodes targeted
        memory_strength     = 0.10,    # C_T: weight of memory-based push
        memory_window       = 8,       # C_T: look-back window in time steps
    )
    regulated = run_regulated(rcfg)

    plot_comparison(baseline, regulated, save_path="figures/comparison.png")

    print("=== Baseline ===")
    print(f"t_exit_V         : {baseline['t_exit_V']}")
    print(f"t_delta_collapse : {baseline['t_delta_collapse']}")
    print(f"returned_to_V    : {baseline['returned_to_V']}")
    print(f"final_delta      : {baseline['final_delta']:.6f}")
    print(f"final_g          : {baseline['final_g']:.6f}")
    print(f"STR              : {int(np.sum(baseline['regime_history'] == 1))}")
    print(f"RSR              : {int(np.sum(baseline['regime_history'] == 2))}")

    print("\n=== Regulated ===")
    print(f"t_exit_V         : {regulated['t_exit_V']}")
    print(f"t_delta_collapse : {regulated['t_delta_collapse']}")
    print(f"returned_to_V    : {regulated['returned_to_V']}")
    print(f"fraction_viable  : {regulated['fraction_viable']:.3f}")
    print(f"final_delta      : {regulated['final_delta']:.6f}")
    print(f"final_g          : {regulated['final_g']:.6f}")
    print(f"STR              : {int(np.sum(regulated['regime_history'] == 1))}")
    print(f"RSR              : {int(np.sum(regulated['regime_history'] == 2))}")

    print("\n=== Key comparison ===")
    print(f"{'Metric':<30} {'Baseline':>12} {'Regulated':>12}")
    print(f"{'fraction_viable':<30} "
          f"{float(np.mean(baseline['viable_history'])):>12.3f} "
          f"{regulated['fraction_viable']:>12.3f}")
    print(f"{'mean_delta (t>20)':<30} "
          f"{float(np.mean(baseline['delta_history'][20:])):>12.6f} "
          f"{float(np.mean(regulated['delta_history'][20:])):>12.6f}")
    print(f"{'mean_g (t>20)':<30} "
          f"{float(np.mean(baseline['g_history'][20:])):>12.6f} "
          f"{float(np.mean(regulated['g_history'][20:])):>12.6f}")
    dh_b = baseline['delta_history']
    dh_r = regulated['delta_history']
    sc_b = int(np.sum(np.diff(np.sign(np.diff(dh_b[20:]))) != 0))
    sc_r = int(np.sum(np.diff(np.sign(np.diff(dh_r[20:]))) != 0))
    print(f"{'delta_sign_changes (t>20)':<30} {sc_b:>12} {sc_r:>12}")
