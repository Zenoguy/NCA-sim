# Memory-Augmented Neural Cellular Automata for KdV Soliton Dynamics

A scientific research prototype investigating whether persistent recurrent memory improves the ability of Neural Cellular Automata (NCAs) to learn, generalize, and stably roll out nonlinear physical dynamics.

---

## 1. Scientific Hypothesis

> **"Does adding persistent memory to a Neural Cellular Automaton (NCA) improve its ability to learn, generalize, and stably roll out nonlinear physical dynamics compared with an otherwise comparable vanilla NCA?"**

This prototype investigates this question on the 1D Korteweg–de Vries (KdV) equation. The study prioritizes falsifiability, reproducibility, and rigorous parameter matching to distinguish true architectural advantages from mere parameter capacity increases.

---

## 2. Mathematical Ground Truth: 1D Korteweg–de Vries Equation

The 1D KdV equation is a classic nonlinear dispersive PDE describing shallow water waves and soliton dynamics:

$$
u_t + \alpha u u_x + \beta u_{xxx} = 0
$$

The prototype standardizes on the canonical normalized form ($\alpha = 6.0, \beta = 1.0$):

$$
u_t + 6 u u_x + u_{xxx} = 0
$$

### Numerical Integrator: ETDRK4
We use a high-precision Fourier pseudospectral spatial discretization with **Exponential Time Differencing 4th-Order Runge-Kutta (ETDRK4)** (Kassam & Trefethen, 2005):
- **Linear dispersion**: $L(k) = i \beta k^3$ in Fourier space, solved analytically via exponential integrating factors to bypass stiff dispersion time-step limits.
- **Nonlinear convection**: $N(u) = -\frac{\alpha}{2} \partial_x (u^2)$, dealiased using the standard $2/3$ rule.
- **Complex contour integration**: Numerical stability for ETDRK4 operators evaluated via 32-point complex circle contours around $z=0$, achieving spectral accuracy down to machine epsilon ($10^{-16}$).

### Analytical Soliton Solutions
The exact traveling 1-soliton solution on the periodic domain is:

$$
u(x, t) = A \operatorname{sech}^2\left(\sqrt{\frac{\alpha A}{12 \beta}} (x - v t - x_0)\right), \quad v = \frac{\alpha A}{3}
$$

### Conserved Quantities
Under periodic boundary conditions, the KdV equation possesses infinite conservation laws. The lowest-order invariants are tracked:
1. **Zeroth-order integral**: $I_1(u) = \int u \, dx$
2. **Quadratic invariant**: $I_2(u) = \int u^2 \, dx$
3. **Hamiltonian**: $I_3(u) = \int \left(\frac{\alpha}{3} u^3 - \beta (u_x)^2\right) dx$

---

## 3. Methodological Invariant: Macro Time vs. Micro-Steps

$$\boxed{\Delta T = \text{constant while } K \text{ varies}}$$
$$\boxed{K \text{ NCA recurrent updates} \equiv \Delta T \text{ physical observation interval}}$$

- The numerical solver advances with internal time steps $\delta t_{\text{solver}} \ll \Delta T$ (e.g. $10^{-3}$) for numerical precision.
- The physical dataset contains observations strictly at macro times: $t_0, t_0+\Delta T, t_0+2\Delta T, \dots$.
- **Only macro states are supervised by the training loss**. Intermediate NCA states are unobserved latent micro-steps.
- Comparisons across $K \in \{1, 2, 4, 8\}$ evaluate on identical $\Delta T$ and identical ground-truth frames, testing whether deeper recurrence per physical time unit improves learning.

---

## 4. Model Architectures & Controls

| Model | Channels | Parameters | Purpose |
|---|---|---|---|
| **Numerical ETDRK4** | - | - | Ground truth reference |
| **Vanilla NCA (Equal-Hidden)** | $C_h=16, C_m=0$ | 5,113 | Standard local baseline |
| **Vanilla NCA (Param-Matched)** | $C_h=23, C_m=0$ | 7,765 | Strict capacity control ($\Delta \approx 0.05\%$) |
| **Memory-NCA (No Persistence)** | $C_h=16, C_m=16$ | 7,769 | Gating architecture control (memory reset every $\Delta T$) |
| **Memory-NCA (Random Persistence)** | $C_h=16, C_m=16$ | 7,769 | Static persistent capacity control ($m_i \sim \mathcal{N}(0, 1)$) |
| **Memory-NCA (Persistent)** | $C_h=16, C_m=16$ | 7,769 | Proposed model: endogenous learned persistent memory |
| **CNN Baseline** | 4-layer circular conv | 10,657 | Non-cellular convolutional surrogate |

### Vanilla NCA
- Strictly local cellular receptive field: radius $r=1$ (kernel size 3) with circular periodic boundary padding.
- State: $s_i = [u_i, h_i] \in \mathbb{R}^{1 + C_h}$.
- Update: $\Delta s_i = F_\theta(\operatorname{Perception}(s_i))$, with residual step $s^{t+1} = s^t + \Delta s^t$.

### Memory-NCA
- Cellular distributed memory: $m_i \in \mathbb{R}^{C_m}$ per cell ($N \times C_m$ total memory values).
- Architectural update order enforced strictly within each micro-step:
  $$\boxed{P_t \longrightarrow m_{t+1} \longrightarrow s_{t+1}}$$
  1. Local perception: $P_t = \operatorname{Perception}(s_t)$
  2. Gated memory update:
     $$\tilde{m}_{t+1} = \tanh(W_m [P_t, m_t] + b_m), \quad g_t = \sigma(W_g [P_t, m_t] + b_g)$$
     $$m_{t+1} = g_t \odot m_t + (1 - g_t) \odot \tilde{m}_{t+1}$$
  3. Physical state update: $\Delta s_t = F_\theta([P_t, m_{t+1}])$, $s_{t+1} = s_t + \Delta s_t$.

---

## 5. Dataset Partitions & Leakage Prevention

Trajectories are partitioned strictly by complete trajectory and parameter configuration to prevent frame leakage:
1. **`train`**: On-manifold solitons with $A \in [0.6, 1.2]$ and randomized $x_0$.
2. **`val`**: Independent random seeds in $A \in [0.6, 1.2]$.
3. **`test_interp`**: Held-out interior parameter combinations within training range.
4. **`test_extrap`**: Soliton amplitudes strictly outside training distribution ($A \in [1.3, 1.8]$).
5. **`test_unseen_params` (Test A)**: Unseen single-soliton parameters.
6. **`test_perturbed_pulses` (Test B)**: Off-manifold pulses with $L \ne \sqrt{12\beta/(\alpha A)}$, producing nonlinear dispersion and radiation.
7. **`test_two_pulses` (Test C)**: Two-pulse collision initial condition ($A_1=1.4, A_2=0.7$) testing nonlinear pass-through and shape preservation.
8. **`test_long_horizon`**: Rollouts up to 100 macro steps ($10\times$ training horizon).

---

## 6. Physics-Aware Metrics

- **Relative $L_2$ Error**: $E_{L2}(t) = \frac{\|\hat{u}(t) - u(t)\|_2}{\|u(t)\|_2}$
- **Peak Amplitude Error**: $|\max_x \hat{u}(x, t) - \max_x u(x, t)|$
- **Centroid Error**: Wave center tracked via circular periodic statistics.
- **FWHM Width Error**: Full width at half maximum difference.
- **Shape Profile Error**: Cosine distance between normalized wave profiles ($1 - \cos(\hat{u}, u)$).
- **Invariant Drifts**: Fractional drift in $I_1, I_2, I_3$.
- **Parallel Scaling Efficiency**: $\text{Efficiency}(p) = \frac{T_1}{p \cdot T_p}$ for $p$ CPU threads.

---

## 7. Reproduction Commands

### Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Stage 1 & 2: Numerical Solver Verification
```bash
python scripts/verify_solver.py
```

### Run All Experiments & Generate Full Report
```bash
python scripts/run_all.py --config configs/default.yaml
```

### Run Individual Experiments
```bash
# Core multi-seed benchmark & Early Falsification Gate
python scripts/run_experiment.py --config configs/default.yaml

# Memory dimension ablation sweep (C_m in 0, 4, 8, 16, 32, 64)
python scripts/run_memory_ablation.py --config configs/default.yaml

# Physical generalization suite (Interpolation, Extrapolation, Tests A, B, C)
python scripts/run_generalization.py --config configs/default.yaml

# K-sensitivity at fixed Delta T
python scripts/run_k_sensitivity.py --config configs/default.yaml

# Causal memory swapping with controls
python scripts/run_memory_swapping.py --config configs/default.yaml

# Multicore CPU scaling benchmark (Ryzen 5 5600H)
python scripts/benchmark_cpu.py --config configs/default.yaml

# Three Controlled Environments Benchmark (Testing Non-Markovian Memory Advantage)
python scripts/run_three_environments.py
```

---

## 8. Key Findings & Early Falsification Gate

1. **Short/Medium Horizon Rollout**:
   - Parameter-matched Vanilla NCA achieved $E_{L2} = 2.52 \pm 1.07$.
   - Persistent Memory-NCA achieved $E_{L2} = 2.89 \pm 1.65$ (14.7% higher error).
   - *Conclusion*: Adding persistent recurrent memory did NOT improve short-to-medium autonomous rollout accuracy on single-soliton KdV flows compared to an equal-parameter Vanilla NCA.
2. **Long-Horizon Autoregressive Stability**:
   - Vanilla NCA without memory experienced extreme compounding numerical explosion on long-horizon rollouts ($E_{L2} \sim 10^{17}$).
   - Persistent Memory-NCA and Random-Persistence Memory-NCA bounded long-horizon divergence substantially ($10^{10} - 10^{13}$, 4 orders of magnitude lower error).
   - Recurrent gating functions as an effective temporal damping / regularization mechanism during extended rollouts.
3. **Hardware Efficiency**:
   - Local cellular NCA updates scale effectively across CPU cores, achieving low single-simulation latency ($< 1$ ms per macro-step on AMD Ryzen 5 5600H).

---

## 9. Three Controlled Environments Benchmark

To test whether the benefit of persistent memory increases when the underlying physical system requires temporal history, we evaluated parameter-matched models across three controlled regimes:

1. **Environment A (Fully Observed KdV)**: Markovian baseline ($u(t) \to u(t+\Delta T)$).
2. **Environment B (Partially Observed KdV)**: Sparse probe stream ($P=16$ probes out of $N=128$, $12.5\%$ spatial coverage). Continuous field reconstruction.
3. **Environment C (Coupled Non-Markovian KdV)**: Mori-Zwanzig system ($u_t + 6 u u_x + u_{xxx} = w, w_t = -\lambda w + \kappa u$). Latent field $w$ is strictly hidden.

### Results Across Environments (Mean $\pm$ Std across Seeds [42, 123]):

| Environment | Model | Parameters | Val Rollout Rel $L_2$ | Memory Advantage (%) |
|---|---|---|---|---|
| **Env A (Fully Observed)** | Vanilla NCA | 7,765 | 3.495e-02 $\pm$ 6.9e-03 | Baseline |
| **Env A (Fully Observed)** | Memory-NCA | 7,769 | 1.185e-01 $\pm$ 1.1e-02 | **-239.0%** |
| **Env B (Sparse Probes)** | Vanilla NCA | 7,765 | 6.955e-01 $\pm$ 2.6e-03 | Baseline |
| **Env B (Sparse Probes)** | Memory-NCA | 7,769 | 7.042e-01 $\pm$ 2.4e-02 | **-1.2%** |
| **Env C (Coupled Memory)** | Vanilla NCA | 7,765 | 4.940e-02 $\pm$ 1.3e-02 | Baseline |
| **Env C (Coupled Memory)** | Memory-NCA | 7,769 | 1.029e-01 $\pm$ 1.4e-02 | **-108.4%** |

*Takeaway*: Because Vanilla NCA retains continuous un-gated hidden channels $h \in \mathbb{R}^{C_h \times N}$ across rollout steps, its residual recurrence ($h \leftarrow h + \Delta h$) already possesses memory capacity. Dedicating parameter capacity to explicit sigmoid gating under matched parameter budgets (~7,765 parameters) reduces the capacity available for spatial perception and nonlinear mixing.

---

## 10. Known Limitations

1. **Single-PDE Scope**: Results are specific to the 1D Korteweg–de Vries equation with periodic boundaries; behavior on dissipative systems (Burgers, Kuramoto–Sivashinsky) may differ.
2. **Local vs. Nonlocal Operators**: Higher-order dispersion ($\partial_{xxx}$) requires multi-step spatial propagation to communicate across cells, making purely local NCAs challenging without sufficient recurrent depth $K$.
3. **Autoregressive Error Accumulation**: Like all recurrent neural surrogates, errors accumulate over long horizons; physical conservation loss penalties may be needed for thousand-step rollouts.

