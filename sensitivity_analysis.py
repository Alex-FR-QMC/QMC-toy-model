import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from Baseline import SimulationConfig, run_batch, summarize_batch

# ============================================================
# Section 4 — Parametric Sensitivity Analysis
# Reproduces: figures/sensitivity.png + printed tables
# ============================================================

os.makedirs("figures", exist_ok=True)

seeds  = list(range(10))

kappas = [0.05, 0.10, 0.15, 0.22, 0.30, 0.40, 0.60]
alphas = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
sigmas = [0.00, 0.01, 0.028, 0.05, 0.08, 0.12, 0.20]


# --------------------------------------------------------
# Helper
# --------------------------------------------------------

def sweep(param: str, values: list) -> list:
    """Run a batch for each value of a single parameter."""
    results = []
    for v in values:
        cfgs   = [SimulationConfig(seed=s, **{param: v}) for s in seeds]
        summary = summarize_batch(run_batch(cfgs))
        results.append(summary)
    return results


def extract(results: list, key: str) -> list:
    return [r[key] for r in results]


# --------------------------------------------------------
# Sweeps
# --------------------------------------------------------

r_kappa = sweep("coupling_strength",  kappas)
r_alpha = sweep("reflexive_strength", alphas)
r_sigma = sweep("noise_std",          sigmas)


# --------------------------------------------------------
# Tables (Section 4.1 / 4.2 / 4.3)
# --------------------------------------------------------

print("=== Section 4.1 — Coupling strength sweep ===")
print(f"{'coupling':>10}  {'mean_t_exit_V':>14}  {'mean_t_delta_collapse':>22}"
      f"  {'mean_t_g_collapse':>18}  {'returned_to_V':>14}")
for v, r in zip(kappas, r_kappa):
    print(f"{v:>10.3f}  {r['mean_t_exit_V']:>14.2f}  {r['mean_t_delta_collapse']:>22.2f}"
          f"  {r['mean_t_g_collapse']:>18.2f}  {r['fraction_returned_to_V']:>14.2f}")

print("\n=== Section 4.2 — Reflexive strength sweep ===")
print(f"{'reflexive':>10}  {'mean_t_exit_V':>14}  {'mean_t_delta_collapse':>22}"
      f"  {'mean_t_g_collapse':>18}  {'returned_to_V':>14}")
for v, r in zip(alphas, r_alpha):
    print(f"{v:>10.3f}  {r['mean_t_exit_V']:>14.2f}  {r['mean_t_delta_collapse']:>22.2f}"
          f"  {r['mean_t_g_collapse']:>18.2f}  {r['fraction_returned_to_V']:>14.2f}")

print("\n=== Section 4.3 — Noise amplitude sweep ===")
print(f"{'noise_std':>10}  {'mean_t_exit_V':>14}  {'mean_t_delta_collapse':>22}"
      f"  {'mean_t_g_collapse':>18}  {'returned_to_V':>14}")
for v, r in zip(sigmas, r_sigma):
    print(f"{v:>10.3f}  {r['mean_t_exit_V']:>14.2f}  {r['mean_t_delta_collapse']:>22.2f}"
          f"  {r['mean_t_g_collapse']:>18.2f}  {r['fraction_returned_to_V']:>14.2f}")

print("\n=== Section 4.4 — Edge case: coupling=0 ===")
print(f"{'kappa':>8}  {'alpha':>8}  {'sigma':>8}  {'mean_t_exit_V':>14}"
      f"  {'returned_to_V':>14}  {'mean_final_delta':>17}")
for alpha in [0.00, 0.05, 0.15]:
    cfgs    = [SimulationConfig(seed=s, coupling_strength=0.0,
                                reflexive_strength=alpha, noise_std=0.05)
               for s in seeds]
    summary = summarize_batch(run_batch(cfgs))
    print(f"{0.0:>8.3f}  {alpha:>8.3f}  {0.05:>8.3f}"
          f"  {summary['mean_t_exit_V']:>14.2f}"
          f"  {summary['fraction_returned_to_V']:>14.2f}"
          f"  {summary['mean_final_delta']:>17.6f}")


# --------------------------------------------------------
# Figure — 2 rows x 3 cols
# Row 1 : mean_t_exit_V     vs kappa / alpha / sigma
# Row 2 : mean_t_delta_collapse vs kappa / alpha / sigma
# --------------------------------------------------------

NOMINAL = [0.22, 0.15, 0.028]
XLABS   = ["coupling strength k", "reflexive strength a", "noise std s"]
XS      = [kappas, alphas, sigmas]
RS      = [r_kappa, r_alpha, r_sigma]
C1, C2  = "#2166ac", "#d6604d"

fig = plt.figure(figsize=(15, 9))
fig.suptitle(
    "QMC Toy Model — Parametric Sensitivity Analysis (mean over 10 seeds)",
    fontsize=13, y=0.99
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.35)

ROWS = [
    ("mean_t_exit_V",          "mean t_exit_V",         C1, "o"),
    ("mean_t_delta_collapse",  "mean t_delta_collapse",  C2, "s"),
]

for row_i, (key, ylabel, color, marker) in enumerate(ROWS):
    for col_i, (xs, rs, nom, xlabel) in enumerate(zip(XS, RS, NOMINAL, XLABS)):
        ax = fig.add_subplot(gs[row_i, col_i])
        ys = extract(rs, key)
        ax.plot(xs, ys, f"{marker}-", color=color, lw=2, ms=6)
        ax.axvline(nom, ls=":", lw=1.3, color="gray", alpha=0.75,
                   label="nominal" if col_i == 0 else None)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(f"{ylabel} vs {xlabel.split()[0]}", fontsize=10)
        ax.grid(True, alpha=0.25)
        if col_i == 0:
            ax.legend(fontsize=8)

fig.text(
    0.5, 0.005,
    "returned_to_V = 0.00 across all non-degenerate configurations  |  "
    "Dotted line = nominal value",
    ha="center", fontsize=10, color="#444444"
)

plt.savefig("figures/sensitivity.png", dpi=300, bbox_inches="tight")
print("\nFigure saved: figures/sensitivity.png")
