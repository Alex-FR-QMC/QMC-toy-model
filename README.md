# QMC Toy Model — Computational Grounding of the QMC Framework

Minimal dynamical model illustrating viability collapse and operational
necessity in the Quantum Meta-Cognition (QMC) paradigm.

---

## Description

This repository implements a discrete tensional field τᵢ(t) defined on a
Watts-Strogatz graph (N=12 nodes) and tracks two morphodynamic observables
derived from Chapter 1 of the QMC paradigm:

- **Δ(t)** — global field dispersion (morphodynamic corridor)
- **𝒢(t)** — mean local contrast over edges (transformable gradient)

The model demonstrates three results:

1. An unregulated distributed tensional field collapses structurally toward
   a homogeneous attractor, exiting the viability domain 𝒱 — regardless of
   parametric configuration or external perturbation.
2. This collapse is faster under FEP-like (gradient descent) dynamics than
   under the QMC unregulated baseline.
3. A minimal set of QMC-inspired mechanisms (anti-synchronisation, drift
   detection, directed perturbation, short local memory) is sufficient to
   maintain viability across the full simulation and across all tested seeds.

---

## Repository structure

| File | Role | Paper section |
|---|---|---|
| `Baseline.py` | Unregulated model + perturbation experiments | 2, 3 |
| `sensitivity_analysis.py` | Parametric sweep | 4 |
| `mcq_regulated.py` | QMC-regulated dynamics | 5 |
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

## Related publications

Quemeneur, A. (2026). *A Morphodynamic Theory of Cognitive Viability:
Foundations of the Quantum Meta-Cognition Paradigm* (Chapter 1).
arXiv preprint. [arXiv link]

Quemeneur, A. (2026). *A Minimal Dynamical Illustration of Viability
in the QMC Framework*. [arXiv link — to be added upon submission]

---

## License

MIT
