result = run_simulation(SimulationConfig(seed=42))
tau_history = result["tau_history"]

for t in range(31, 300):
    recent_delta = np.std(tau_history[t-20:t])
    dtau_approx = np.mean(np.abs(np.diff(tau_history[t-5:t], axis=0)))
    if t % 30 == 0:
        print(f"t={t} |
recent_delta={recent_delta:.4f} |
recent_drift={dtau_approx:.4f}")
