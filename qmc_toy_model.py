import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass
from typing import Optional, Dict, Any, List


# ============================================================
# Configuration
# ============================================================

@dataclass
class SimulationConfig:
    T: int = 300
    N: int = 12
    dt: float = 0.08

    # Graph
    graph_mode: str = "ws"   # "ws", "erdos", "ring", "complete"
    ws_k: int = 4
    ws_p: float = 0.3
    erdos_p: float = 0.25

    # Dynamics
    coupling_strength: float = 0.22
    reflexive_strength: float = 0.15
    noise_std: float = 0.028

    # Initial field
    tau_min_init: float = 0.35
    tau_max_init: float = 0.65

    # Hard clipping
    tau_clip_min: float = 0.09
    tau_clip_max: float = 0.91

    # Viability thresholds
    delta_crit: float = 0.32
    g_min: float = 0.045

    # Collapse thresholds for analysis
    delta_floor: float = 0.01
    g_floor: float = 0.01

    # Regime detection thresholds (calibrated empirically)
    str_drift_max: float = 0.002
    str_delta_min: float = 0.001
    str_delta_max: float = 0.05
    rsr_delta_min: float = 0.05

    # Shock perturbation
    shock_time: Optional[int] = None
    shock_node: Optional[int] = None
    shock_amplitude: float = 0.0

    # Structural perturbation
    rewire_time: Optional[int] = None
    rewire_mode: Optional[str] = None  # "random_rewire", "densify", "sparsify"
    rewire_fraction: float = 0.15

    # Random seed
    seed: int = 42


# ============================================================
# Graph utilities
# ============================================================

def make_graph(cfg: SimulationConfig) -> nx.Graph:
    if cfg.graph_mode == "ws":
        G = nx.watts_strogatz_graph(cfg.N, k=cfg.ws_k, p=cfg.ws_p, seed=cfg.seed)
    elif cfg.graph_mode == "erdos":
        G = nx.erdos_renyi_graph(cfg.N, p=cfg.erdos_p, seed=cfg.seed)
    elif cfg.graph_mode == "ring":
        G = nx.watts_strogatz_graph(cfg.N, k=cfg.ws_k, p=0.0, seed=cfg.seed)
    elif cfg.graph_mode == "complete":
        G = nx.complete_graph(cfg.N)
    else:
        raise ValueError(f"Unknown graph_mode: {cfg.graph_mode}")

    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        G = nx.convert_node_labels_to_integers(G)

    return G


def adjacency_matrix(G: nx.Graph, N_expected: int) -> np.ndarray:
    A = nx.to_numpy_array(G)
    if A.shape[0] != N_expected:
        raise ValueError(
            f"Graph size mismatch: expected {N_expected}, got {A.shape[0]}."
        )
    return A


def rewire_graph(G: nx.Graph, cfg: SimulationConfig, rng: np.random.Generator) -> nx.Graph:
    G_new = G.copy()
    edges = list(G_new.edges())
    non_edges = list(nx.non_edges(G_new))

    if cfg.rewire_mode == "random_rewire":
        n_rewire = max(1, int(len(edges) * cfg.rewire_fraction))
        edges_to_remove = rng.choice(len(edges), size=min(n_rewire, len(edges)), replace=False)
        G_new.remove_edges_from([edges[i] for i in edges_to_remove])
        non_edges = list(nx.non_edges(G_new))
        if len(non_edges) > 0:
            add_idx = rng.choice(len(non_edges), size=min(n_rewire, len(non_edges)), replace=False)
            G_new.add_edges_from([non_edges[i] for i in add_idx])

    elif cfg.rewire_mode == "densify":
        n_add = max(1, int(len(non_edges) * cfg.rewire_fraction))
        if len(non_edges) > 0:
            add_idx = rng.choice(len(non_edges), size=min(n_add, len(non_edges)), replace=False)
            G_new.add_edges_from([non_edges[i] for i in add_idx])

    elif cfg.rewire_mode == "sparsify":
        n_remove = max(1, int(len(edges) * cfg.rewire_fraction))
        if len(edges) > 0:
            rem_idx = rng.choice(len(edges), size=min(n_remove, len(edges)), replace=False)
            G_new.remove_edges_from([edges[i] for i in rem_idx])

    else:
        raise ValueError(f"Unknown rewire_mode: {cfg.rewire_mode}")

    if not nx.is_connected(G_new):
        largest_cc = max(nx.connected_components(G_new), key=len)
        G_new = G_new.subgraph(largest_cc).copy()
        G_new = nx.convert_node_labels_to_integers(G_new)
        if G_new.number_of_nodes() != cfg.N:
            return G.copy()

    return G_new


# ============================================================
# Observables
# ============================================================

def compute_delta(tau: np.ndarray) -> float:
    return float(np.std(tau))


def compute_g_on_edges(tau: np.ndarray, A: np.ndarray) -> float:
    edge_diffs: List[float] = []
    n = len(tau)
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] > 0:
                edge_diffs.append(abs(tau[i] - tau[j]))
    return float(np.mean(edge_diffs)) if edge_diffs else 0.0


def in_viability(delta: float, g: float, cfg: SimulationConfig) -> bool:
    return (0.0 < delta < cfg.delta_crit) and (g > cfg.g_min)


# ============================================================
# Regime detection
# 0 = unclassified / early transient
# 1 = STR (Stationary Tension Regime)
# 2 = RSR (Resonant / oscillatory regime)
# ============================================================

def detect_regime(
    t: int,
    tau_history: np.ndarray,
    dtau: np.ndarray,
    cfg: SimulationConfig,
    window: int = 20
) -> int:
    if t < window + 1:
        return 0

    recent_delta = float(np.std(tau_history[t - window:t]))
    recent_drift = float(np.mean(np.abs(dtau)))

    if (recent_drift < cfg.str_drift_max
            and cfg.str_delta_min < recent_delta < cfg.str_delta_max):
        return 1  # STR

    if recent_delta > cfg.rsr_delta_min:
        return 2  # RSR

    return 0


# ============================================================
# Metrics
# ============================================================

def first_time_below(series: np.ndarray, threshold: float) -> Optional[int]:
    idx = np.where(series < threshold)[0]
    return int(idx[0]) if len(idx) > 0 else None


def first_permanent_exit(viable_history: np.ndarray) -> Optional[int]:
    T = len(viable_history)
    for t in range(T):
        if not viable_history[t] and np.all(~viable_history[t:]):
            return t
    return None


def returned_to_viability(viable_history: np.ndarray) -> bool:
    ever_false = False
    for v in viable_history:
        if not v:
            ever_false = True
        if ever_false and v:
            return True
    return False


# ============================================================
# Main simulation
# ============================================================

def run_simulation(cfg: SimulationConfig) -> Dict[str, Any]:
    rng = np.random.default_rng(cfg.seed)

    G = make_graph(cfg)
    A = adjacency_matrix(G, cfg.N)

    tau = np.linspace(cfg.tau_min_init, cfg.tau_max_init, cfg.N)

    tau_history  = np.zeros((cfg.T, cfg.N))
    delta_history = np.zeros(cfg.T)
    g_history     = np.zeros(cfg.T)
    viable_history = np.zeros(cfg.T, dtype=bool)
    regime_history = np.zeros(cfg.T, dtype=int)

    dtau = np.zeros(cfg.N)

    for t in range(cfg.T):
        # Structural perturbation
        if cfg.rewire_time is not None and cfg.rewire_mode is not None and t == cfg.rewire_time:
            G = rewire_graph(G, cfg, rng)
            A = adjacency_matrix(G, cfg.N)

        # Shock perturbation
        if cfg.shock_time is not None and t == cfg.shock_time:
            shock_node = (cfg.shock_node if cfg.shock_node is not None
                          else int(rng.integers(0, cfg.N)))
            tau[shock_node] += cfg.shock_amplitude
            tau = np.clip(tau, cfg.tau_clip_min, cfg.tau_clip_max)

        # Observables
        delta = compute_delta(tau)
        g     = compute_g_on_edges(tau, A)
        is_viable = in_viability(delta, g, cfg)

        delta_history[t]  = delta
        g_history[t]      = g
        viable_history[t] = is_viable
        regime_history[t] = detect_regime(t, tau_history, dtau, cfg)

        # Dynamics
        gamma_pp = rng.normal(0.0, cfg.noise_std, cfg.N)
        degree   = np.sum(A, axis=1)
        coupling  = cfg.coupling_strength * (A @ tau - degree * tau)
        reflexive = -cfg.reflexive_strength * (tau - np.mean(tau))

        dtau = cfg.dt * (coupling + reflexive + 0.48 * gamma_pp)
        tau  = np.clip(tau + dtau, cfg.tau_clip_min, cfg.tau_clip_max)

        tau_history[t] = tau.copy()

    return {
        "config":         cfg,
        "graph":          G,
        "adjacency":      A,
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
    }


# ============================================================
# Plotting — 3x2 grid
# Row 1: Field trajectories | Morphodynamic corridor
# Row 2: Gradient           | Viability
# Row 3: Observed regimes (full width)
# ============================================================

def plot_simulation(
    result: Dict[str, Any],
    title: str = "MCQ Toy Model — Baseline (non-regulated)",
    save_path: Optional[str] = None
) -> None:
    cfg            = result["config"]
    tau_history    = result["tau_history"]
    delta_history  = result["delta_history"]
    g_history      = result["g_history"]
    viable_history = result["viable_history"]
    regime_history = result["regime_history"]
    timesteps      = np.arange(cfg.T)

    fig = plt.figure(figsize=(14, 13))
    fig.suptitle(title, fontsize=16, y=0.98)

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.32)

    ax_field   = fig.add_subplot(gs[0, 0])
    ax_delta   = fig.add_subplot(gs[0, 1])
    ax_g       = fig.add_subplot(gs[1, 0])
    ax_viable  = fig.add_subplot(gs[1, 1])
    ax_regime  = fig.add_subplot(gs[2, :])   # full width

    # --- Panel 1: Field trajectories ---
    for i in range(cfg.N):
        ax_field.plot(tau_history[:, i], lw=1.4)
    ax_field.set_title("Evolution of Tensional Field τᵢ(t)")
    ax_field.set_xlabel("Time steps")
    ax_field.set_ylabel("τᵢ")

    # --- Panel 2: Morphodynamic corridor ---
    ax_delta.plot(delta_history, lw=2, label="Δ(t)")
    ax_delta.axhline(cfg.delta_crit,  ls="--", lw=1.5, color="red",  label="Δ_crit")
    ax_delta.axhline(cfg.delta_floor, ls=":",  lw=1.2, color="gray", label="Δ_floor")
    ax_delta.set_title("Morphodynamic Corridor Δ(t)")
    ax_delta.set_xlabel("Time steps")
    ax_delta.set_ylabel("Δ")
    ax_delta.legend()

    # --- Panel 3: Discrete transformable gradient ---
    ax_g.plot(g_history, lw=2, color="darkorange", label="G(t)")
    ax_g.axhline(cfg.g_min,   ls="--", lw=1.5, color="red",  label="G_min")
    ax_g.axhline(cfg.g_floor, ls=":",  lw=1.2, color="gray", label="G_floor")
    ax_g.set_title("Discrete Transformable Gradient G(t)")
    ax_g.set_xlabel("Time steps")
    ax_g.set_ylabel("G")
    ax_g.legend()

    # --- Panel 4: Viability ---
    ax_viable.plot(viable_history.astype(int), lw=2, color="green")
    ax_viable.set_ylim(-0.1, 1.1)
    ax_viable.set_yticks([0, 1])
    ax_viable.set_yticklabels(["No", "Yes"])
    ax_viable.set_title("Membership in Viability Domain V")
    ax_viable.set_xlabel("Time steps")
    ax_viable.set_ylabel("x(t) ∈ V")

    # --- Panel 5: Observed regimes (full width) ---
    str_times = timesteps[regime_history == 1]
    rsr_times = timesteps[regime_history == 2]

    if len(str_times) > 0:
        ax_regime.scatter(str_times, np.zeros(len(str_times)),
                          s=6, color="blue", label="STR (Stationary)", zorder=3)
    if len(rsr_times) > 0:
        ax_regime.scatter(rsr_times, np.ones(len(rsr_times)),
                          s=6, color="orange", label="RSR (Resonant)", zorder=3)

    ax_regime.set_xlim(0, cfg.T)
    ax_regime.set_ylim(-0.5, 1.5)
    ax_regime.set_yticks([0, 1])
    ax_regime.set_yticklabels(["STR", "RSR"])
    ax_regime.set_title("Observed Dynamic Regimes")
    ax_regime.set_xlabel("Time steps")
    ax_regime.legend(loc="upper right", markerscale=3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


# ============================================================
# Batch experiments
# ============================================================

def run_batch(configs: List[SimulationConfig]) -> List[Dict[str, Any]]:
    return [run_simulation(cfg) for cfg in configs]


def summarize_batch(results: List[Dict[str, Any]]) -> Dict[str, float]:
    def valid_vals(key: str):
        vals = [r[key] for r in results if r[key] is not None]
        return np.array(vals, dtype=float) if vals else np.array([])

    exit_vals  = valid_vals("t_exit_V")
    delta_vals = valid_vals("t_delta_collapse")
    g_vals     = valid_vals("t_g_collapse")

    return {
        "n_runs":                   len(results),
        "mean_final_delta":         float(np.mean([r["final_delta"] for r in results])),
        "std_final_delta":          float(np.std([r["final_delta"]  for r in results])),
        "mean_final_g":             float(np.mean([r["final_g"]     for r in results])),
        "std_final_g":              float(np.std([r["final_g"]      for r in results])),
        "fraction_returned_to_V":   float(np.mean([r["returned_to_V"] for r in results])),
        "mean_t_exit_V":            float(np.mean(exit_vals))  if len(exit_vals)  > 0 else float("nan"),
        "mean_t_delta_collapse":    float(np.mean(delta_vals)) if len(delta_vals) > 0 else float("nan"),
        "mean_t_g_collapse":        float(np.mean(g_vals))     if len(g_vals)     > 0 else float("nan"),
    }


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)

    # --------------------------------------------------------
    # 1) Baseline run
    # --------------------------------------------------------
    cfg    = SimulationConfig(seed=42)
    result = run_simulation(cfg)
    plot_simulation(
        result,
        title="QMC Toy Model — Baseline (non-regulated)",
        save_path="figures/baseline.png"
    )

    print("=== Baseline summary ===")
    print(f"t_exit_V           : {result['t_exit_V']}")
    print(f"t_delta_collapse   : {result['t_delta_collapse']}")
    print(f"t_g_collapse       : {result['t_g_collapse']}")
    print(f"returned_to_V      : {result['returned_to_V']}")
    print(f"final_delta        : {result['final_delta']:.6f}")
    print(f"final_g            : {result['final_g']:.6f}")
    print(f"STR detections     : {int(np.sum(result['regime_history'] == 1))}")
    print(f"RSR detections     : {int(np.sum(result['regime_history'] == 2))}")

    # --------------------------------------------------------
    # 2) Noise sensitivity
    # --------------------------------------------------------
    seeds        = list(range(10))
    noise_levels = [0.01, 0.028, 0.05, 0.08]

    print("\n=== Noise sensitivity ===")
    for sigma in noise_levels:
        cfgs    = [SimulationConfig(seed=s, noise_std=sigma) for s in seeds]
        batch   = run_batch(cfgs)
        summary = summarize_batch(batch)
        print(
            f"noise_std={sigma:.3f} | "
            f"mean_t_exit_V={summary['mean_t_exit_V']:.2f} | "
            f"mean_t_delta_collapse={summary['mean_t_delta_collapse']:.2f} | "
            f"mean_t_g_collapse={summary['mean_t_g_collapse']:.2f} | "
            f"returned_to_V={summary['fraction_returned_to_V']:.2f}"
        )

    # --------------------------------------------------------
    # 3) Point perturbation (shock)
    # --------------------------------------------------------
    shock_cfg    = SimulationConfig(seed=42, shock_time=80, shock_node=3, shock_amplitude=0.25)
    shock_result = run_simulation(shock_cfg)
    plot_simulation(
        shock_result,
        title="QMC Toy Model — Point Perturbation",
        save_path="figures/shock.png"
    )

    # --------------------------------------------------------
    # 4) Structural perturbation (rewiring)
    # --------------------------------------------------------
    rewire_cfg    = SimulationConfig(seed=42, rewire_time=60, rewire_mode="random_rewire", rewire_fraction=0.20)
    rewire_result = run_simulation(rewire_cfg)
    plot_simulation(
        rewire_result,
        title="QMC Toy Model — Structural Perturbation",
        save_path="figures/rewire.png"
    )

