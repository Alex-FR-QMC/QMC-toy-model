import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

np.random.seed(42)
T = 300
N = 12
dt = 0.08

G = nx.watts_strogatz_graph(N, k=4, p=0.3)
adj = nx.to_numpy_array(G)

tau = np.linspace(0.35, 0.65, N)
tau_history = np.zeros((T, N))
tau_history[0] = tau.copy()

delta_history = np.zeros(T)
g_history = np.zeros(T)
viable_history = np.zeros(T, dtype=bool)
regime_history = np.zeros(T, dtype=int)  # 0=STR, 1=RSR

delta_crit = 0.32
g_min = 0.045

for t in range(T):
    delta = np.std(tau)
    delta_history[t] = delta
    
    edges = list(G.edges())
if edges:
    g = np.mean([np.abs(tau[i] - tau[j]) for i, j in edges])
else:
    g = 0.0
g_history[t] = g
    
    in_v = (delta < delta_crit) and (g > g_min)
    viable_history[t] = in_v
    
    gamma_pp = np.random.normal(0, 0.028, N)
    coupling = 0.22 * (adj @ tau - np.sum(adj, axis=1) * tau)
    reflexive = -0.15 * (tau - np.mean(tau))
    
    dtau = dt * (0.18 * g * (1 - delta/delta_crit) + coupling + reflexive + 0.48 * gamma_pp)
    tau += dtau
    tau = np.clip(tau, 0.09, 0.91)
    tau_history[t] = tau.copy()
    
    if t > 30:
        recent_delta = np.std(tau_history[t-20:t])
        recent_drift = np.mean(np.abs(dtau))
        if recent_drift < 0.008 and recent_delta > 0.022:
            regime_history[t] = 0   # STR
        elif recent_delta > 0.095:
            regime_history[t] = 1   # RSR

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('QMC Toy Model – Chapter 1 Realization (HTS discrete, 12 regions)', fontsize=16)

plt.tight_layout()
plt.savefig('qmc_toy_v2_publication.png', dpi=300)
plt.show()
