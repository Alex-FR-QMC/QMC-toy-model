result = run_simulation(SimulationConfig(seed=42))
tau_history = result["tau_history"]
cfg = SimulationConfig(seed=42)

dt = cfg.dt
for t in range(31, 300):
    recent_delta = np.std(tau_history[t-20:t])
    dtau_exact = (tau_history[t] - tau_history[t-1]) / dt
    recent_drift = float(np.mean(np.abs(dtau_exact)))
    if t % 30 == 0:
        print(f"t={t} | recent_delta={recent_delta:.4f} | recent_drift={recent_drift:.4f}")
