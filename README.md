# Viability Collapse in Distributed Tensional Fields — Computational Study

Minimal dynamical model demonstrating that the structural conditions for
non-trivial transformation in a distributed field are not self-maintaining,
and that a minimal regulatory architecture is sufficient to restore them.

Associated paper: *Viability Collapse in Distributed Tensional Fields: A
Computational Study* (submitted to AI, MDPI).

---

## Description

This repository implements a discrete scalar field τᵢ(t) defined on a
Watts-Strogatz graph (N=12 nodes) and tracks two observable conditions
for viable distributed dynamics:

- **Δ(t)** — global field dispersion, measuring the integrability corridor
- **𝒢(t)** — mean local contrast over graph edges, measuring local
  transformability

A system is *viable* when both conditions are simultaneously satisfied:

```
0 < Δ(t) < Δ_crit    and    𝒢(t) > 𝒢_min
```

The model demonstrates three results:

1. An unregulated distributed field collapses structurally toward a
   homogeneous attractor — permanently exiting the viability domain —
   regardless of parametric configuration or external perturbation.
2. FEP-like (gradient descent) dynamics collapse faster than the
   unregulated baseline, for the same structural reason.
3. A minimal set of four regulatory mechanisms (anti-synchronisation,
   drift detection, directed perturbation, short local memory) is
   sufficient to maintain viability across the full simulation and
   across all tested seeds (fraction_viable = 1.000, 20 seeds, T=300
   and T=1000).

The theoretical grounding of the viability conditions is provided in [4]
(see paper references).

---

## Repository structure

| File | Role | Paper section |
|---|---|---|
| `Baseline.py` | Unregulated model + perturbation experiments | 2, 3 |
| `sensitivity_analysis.py` | Parametric sweep | 4 |
| `qmc_regulated.py` | Regulated dynamics + comparison figure | 5 |
| `fep_comparison.py` | FEP-like comparison | 6 |
| `calibrate_regime_thresholds.py` | STR/RSR threshold justification | Appendix |
| `calibrate_nominal_parameters.py` | Nominal parameter justification | Appendix |
| `robustness_checks.py` | Seed and long-run robustness | Appendix |

---

## Usage

```bash
# Reproduce all paper figures and tables, in order
python Baseline.py
python sensitivity_analysis.py
python mcq_regulated.py
python fep_comparison.py

# Methodological appendix
python calibrate_regime_thresholds.py
python calibrate_nominal_parameters.py
python robustness_checks.py
```

All figures are saved to `figures/`.
All simulations are fully deterministic given their explicit random seed.

---

## Key results at a glance

| System | fraction_viable | t_exit_V | final_delta | final_g |
|---|---|---|---|---|
| FEP-like | 0.023 | 7 | 0.002 | 0.003 |
| Unregulated baseline | 0.030 | 9 | 0.003 | 0.004 |
| Regulated | **1.000** | **None** | **0.039** | **0.055** |

---

## Dependencies

```
Python 3.x
numpy
networkx
matplotlib
```

Install with:

```bash
pip install numpy networkx matplotlib
```

---

## License

MIT
