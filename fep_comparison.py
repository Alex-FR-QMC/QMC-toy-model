"""
fep_comparison.py
=================
Section 6 -- Comparison with Optimization-Based Dynamics

Implements a FEP-like (Free Energy Principle) gradient descent dynamics
on the same graph structure as the QMC toy model, using identical
parameters (coupling, noise, topology, seed).

FEP-like dynamics:
    d_t tau_i = -k*(tau_i - tau*) + k * sum_j A_ij*(tau_j - tau_i) + Gamma_i(t)

    where tau* = 0.5 is the convergence target (analogue of the free
    energy minimum), the first term drives global convergence, and
    the second minimises local discrepancy between neighbours.

Produces a 3-way comparison of Delta(t):
    FEP-like (gradient descent)    -- red
    QMC baseline (unregulated)     -- grey
    QMC regulated                  -- blue

Requires: Baseline.py and mcq_regulated.py in the same directory.

Usage:
    python fep_comparison.py

Outputs:
    figures/fep_comparison.png  -- 3-way Delta(t) comparison
    printed comparison table    -- key metrics for all three systems
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import importlib.util as ilu

from Baseline import (
    SimulationConfig, run_simulation, first_permanent_exit,
    make_graph, adjacency_matrix, compute_delta,
    compute_g_on_edges, in_viability
)
from mcq_regulated import RegulatedConfig, run_regulated


# ============================================================
# FEP-like dynamics
# ============================================================

def run_fep_like(cfg: SimulationConfig, tau_target: float = 0.5) -> dict:
    """
    Gradient descent toward tau_target + local synchronisation.

    Approximates FEP: minimise a global free energy
        F = sum_i (tau_i - tau_target)^2
    subject to local coupling constraints.
    """
    rng = np.random.default_rng(cfg.seed)
    G   = make_graph(cfg)
    A   = adjacency_matrix(G, cfg.N)
    tau = rng.uniform(cfg.tau_min_init, cfg.tau_max_init, cfg.N)

    tau_history    = np.zeros((cfg.T, cfg.N))
    delta_history  = np.zeros(cfg.T)
    g_history      = np.zeros(cfg.T)
    viable_history = np.zeros(cfg.T, dtype=bool)

    for t in range(cfg.T):
        delta     = compute_delta(tau)
        g         = compute_g_on_edges(tau, A)
        is_viable = in_viability(delta, g, cfg)

        delta_history[t]  = delta
        g_history[t]      = g
        viable_history[t] = is_viable
        tau_history[t]    = tau.copy()

        gamma_pp  = rng.normal(0.0, cfg.noise_std, cfg.N)
        degree    = np.sum(A, axis=1)

        # Global convergence: minimise prediction error (FEP term)
        fep_global = -cfg.coupling_strength * (tau - tau_target)

        # Local synchronisation: minimise neighbour discrepancy
        fep_local  = cfg.coupling_strength * (A @ tau - degree * tau)

        dtau = cfg.dt * (fep_global + fep_local + 0.48 * gamma_pp)
        tau  = np.clip(tau + dtau, cfg.tau_clip_min, cfg.tau_clip_max)

    return {
        "tau_history":     tau_history,
        "delta_history":   delta_history,
        "g_history":       g_history,
        "viable_history":  viable_history,
        "fraction_viable": float(np.mean(viable_history)),
        "final_delta":     float(delta_history[-1]),
        "final_g":         float(g_history[-1]),
        "t_exit_V":        first_permanent_exit(viable_history),
    }


# ============================================================
# 3-way comparison figure
# ============================================================

def plot_fep_comparison(
    fep_result:       dict,
    baseline_result:  dict,
    regulated_result: dict,
    cfg:              SimulationConfig,
    save_path:        str = "figures/fep_comparison.png"
) -> None:
    """
    Single-panel figure: Delta(t) for FEP-like, QMC baseline, QMC regulated.
    """
    ts   = np.arange(cfg.T)
    dh_f = fep_result["delta_history"]
    dh_b = baseline_result["delta_history"]
    dh_r = regulated_result["delta_history"]

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(ts, dh_f, lw=2.2, color="#d6604d",
            label="FEP-like (gradient descent)", zorder=3)
    ax.plot(ts, dh_b, lw=2.0, color="#aaaaaa",
            label="QMC baseline (unregulated)", zorder=2)
    ax.plot(ts, dh_r, lw=2.2, color="#2166ac",
            label="QMC regulated", zorder=4)

    ax.axhline(cfg.delta_crit,  ls="--", lw=1.5, color="red",
               label=f"delta_crit = {cfg.delta_crit}")
    ax.axhline(cfg.delta_floor, ls=":",  lw=1.2, color="#888888",
               label=f"delta_floor = {cfg.delta_floor}")

    ax.set_title(
        "Morphodynamic Corridor Delta(t) -- FEP-like vs QMC Baseline vs QMC Regulated",
        fontsize=12
    )
    ax.set_xlabel("Time steps")
    ax.set_ylabel("Delta(t)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)

    cfg      = SimulationConfig(seed=42)
    baseline = run_simulation(cfg)
    fep      = run_fep_like(cfg)

    rcfg = RegulatedConfig(
        base                = SimulationConfig(seed=42),
        anti_sync_strength  = 0.35,
        anti_sync_threshold = 0.12,
        delta_alert         = 0.07,
        directed_strength   = 0.28,
        directed_fraction   = 0.40,
        memory_strength     = 0.10,
        memory_window       = 8,
    )
    regulated = run_regulated(rcfg)

    # --------------------------------------------------------
    # Figure -- Section 6.1
    # --------------------------------------------------------
    plot_fep_comparison(fep, baseline, regulated, cfg,
                        save_path="figures/fep_comparison.png")
    print("Figure saved: figures/fep_comparison.png")

    # --------------------------------------------------------
    # Comparison table -- Section 6
    # --------------------------------------------------------
    dh_b = baseline["delta_history"]
    dh_f = fep["delta_history"]

    t_b = next((t for t, d in enumerate(dh_b) if d < 0.01), None)
    t_f = next((t for t, d in enumerate(dh_f) if d < 0.01), None)

    print("\n=== Section 6 -- 3-way comparison ===")
    print(f"{'Metric':<28} {'FEP-like':>12} {'QMC baseline':>14} {'QMC regulated':>14}")
    print(f"{'fraction_viable':<28} "
          f"{fep['fraction_viable']:>12.3f} "
          f"{float(np.mean(baseline['viable_history'])):>14.3f} "
          f"{regulated['fraction_viable']:>14.3f}")
    print(f"{'t_exit_V':<28} "
          f"{str(fep['t_exit_V']):>12} "
          f"{str(baseline['t_exit_V']):>14} "
          f"{str(regulated['t_exit_V']):>14}")
    print(f"{'final_delta':<28} "
          f"{fep['final_delta']:>12.4f} "
          f"{baseline['final_delta']:>14.4f} "
          f"{regulated['final_delta']:>14.4f}")
    print(f"{'final_g':<28} "
          f"{fep['final_g']:>12.4f} "
          f"{baseline['final_g']:>14.4f} "
          f"{regulated['final_g']:>14.4f}")
    print(f"{'t_delta < 0.01':<28} "
          f"{str(t_f):>12} "
          f"{str(t_b):>14} "
          f"{'None':>14}")
