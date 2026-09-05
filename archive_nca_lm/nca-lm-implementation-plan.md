# NCA-LM Implementation Plan
### What can NCA bring to generative modeling / deep learning that conventional architectures don't naturally provide?

Author's context: continuation of the KdV-NCA project. This plan operationalizes the experimental ladder into a concrete, buildable research codebase, with dataset choices, architecture specs, hyperparameters, compute budgets, and — critically — stop/go gates so this doesn't turn into a second KdV-style detour.

---

## 0. Guiding Philosophy & Repositioning

The KdV project taught us an invaluable scientific lesson: **the most interesting result wasn't necessarily the one we anticipated**, and chasing a pre-ordained advantage (e.g., parameter capacity masquerading as architectural superiority) leads to dead ends.

We do **not** begin by declaring what NCA is "useful for" (e.g., declaring robustness or bounded streaming as our pre-baked niche). Instead, we treat the NCA as an unfamiliar computational primitive and systematically characterize its behavior.

### Core Theoretical Principles:
1. **The Core Architectural Knob is Computation Depth with a Shared Rule:**
   $$\boxed{ \text{same parameters} + \text{same learned rule} + \text{more computation} \implies \text{different quality} }$$
   In conventional architectures, compute is tied to parameters (more layers = more weights). In SSMs (Mamba), state updates happen once per token. In NCA, the micro-step iteration count $K$ is an independent, test-time adjustable computation axis.
2. **The Shared-Rule Test is Essential:**
   A true NCA applies the **same local rule repeatedly**:
   $$s_{k+1} = F_\theta^{(d_k)}(s_k)$$
   If we use different convolution weights at every step ($F_{\theta_1} \to F_{\theta_2} \dots$), we haven't built an NCA — we have simply built a conventional multi-layer dilated CNN. We must explicitly test **Shared NCA Rule** vs. **Step-Specific CNN Stack** to determine whether cellular iteration itself does useful work.
3. **SSMs Already Own Bounded Streaming Memory:**
   State-space models (Mamba/Mamba-2/Mamba-3) and RWKV already provide linear-time, constant-memory sequence modeling. Claiming "$O(1)$ memory" is not novel territory. Our focus is on the iterative local dynamics that SSMs do not have.
4. **Causality at Every Micro-Step:**
   A local convolution iterated $K$ times leaks future tokens backwards unless strictly causal. Left-padding must be enforced on every single micro-step.
5. **Receptive Field Scaling via Dilation:**
   A static radius $r=2$ repeated $K=4$ times only sees 8 tokens. To model language, we use an exponential dilation schedule ($d_k = 2^k$), scaling the receptive field to $\ge 126$ tokens at $K=6$ without adding parameters.

---

## 1. The Research Framework: Four Questions & Success Criteria

The research is structured around four consecutive questions:

```
[Q1: Can it function as an LM?]
           ↓ (Yes)
[Q2: What does its computation profile look like vs. baselines?]
           ↓ (Characterized)
[Q3: Does NCA exhibit an unusual / unpredicted capability?]
           ↓ (Identified)
[Q4: Can that capability improve a conventional model (NCA as component)?]
```

### The Primary Success Criterion:
> **The project succeeds if it identifies at least one reproducible computational behavior of NCA that is not adequately explained by an existing architecture and that creates a measurable advantage on a relevant task.**

Candidate manifestations to probe (none pre-declared as winners):
- **Computation-Depth Scaling:** Does increasing $K$ at inference systematically recover quality on complex, high-entropy, or corrupted tokens?
- **Context & Receptive Field Independence:** How does quality behave when computation ($K$), receptive field ($r \cdot \sum d_k$), and context length ($T$) are independently varied?
- **Perturbation Recovery & Homeostasis:** Mid-sequence hidden state disruption — does the local iterative rule stabilize representations faster than recurrent or feedforward states?
- **Representational Robustness:** Resistance to character/subword noise (typos, omissions).
- **Streaming Activation Dynamics:** Constant-memory activation caching behavior under long rollouts.

---

## 2. Repo Structure

```
nca-lm/
├── configs/
│   ├── level0_ngram.yaml
│   ├── level1_gru.yaml
│   ├── level1_transformer.yaml
│   ├── level1_transformer_sliding.yaml   # context-matched sliding window control
│   ├── level1_mamba.yaml                 # SSM baseline
│   ├── level2_nca_shared.yaml            # true NCA: weight-tied rule F_theta
│   ├── level2_nca_unshared.yaml          # control: step-specific CNN F_{theta_k}
│   ├── level3_hybrid.yaml                # NCA as component / adaptor
│   └── ablations/
│       ├── depth_k_scaling.yaml          # test-time compute sweep
│       └── context_decoupling.yaml       # RF vs compute vs context
├── data/
│   ├── prepare_wikitext2.py
│   ├── prepare_wikitext103.py
│   └── tokenizer.py                      # byte-level BPE, 8k-16k vocab, shared across all models
├── models/
│   ├── ngram.py
│   ├── rnn_baseline.py
│   ├── transformer_baseline.py           # full causal + sliding-window modes
│   ├── mamba_baseline.py                 # reference SSM
│   ├── nca_lm.py                         # core model (supports shared vs unshared rules)
│   └── hybrid.py                         # Transformer + NCA adaptor
├── train.py                              # single unified training entrypoint
├── eval/
│   ├── perplexity.py
│   ├── test_time_compute.py              # evaluate quality vs inference K
│   ├── context_sweep.py                  # evaluate RF vs context decoupling
│   ├── state_disruption.py               # mid-stream perturbation & recovery
│   └── noise_robustness.py               # typo / subword corruption harness
├── notebooks/
│   └── param_matching.ipynb              # strictly assert param parity (target ±5%)
└── README.md                             # append-only decision and results log
```

---

## 3. Shared Infrastructure & Controls

- **Datasets:**
  - **WikiText-2** (~2M tokens) — rapid architecture debugging, calibration, and parameter-matched comparisons.
  - **WikiText-103** (~100M tokens) — confirmation runs for selected models only.
- **Tokenizer:** Byte-level BPE (8k–16k vocab), trained once and shared across every single experiment.
- **Parameter Matching Harness:** Strict assertion script ensuring all primary baselines match target parameter budget (e.g., 10M $\pm 5\%$).
- **Base Context Window:** $T=128$ for initial training ladder; evaluated out to $T \in \{128, 256, 512, 1024\}$.

---

## 4. Question 1 — Can an NCA Function as a Language Model?

**Objective:** Standard autoregressive next-token prediction:
$$P(x_t \mid x_{<t})$$
No architectural cheating, no future leakage.

### Baseline Calibration (Floor & Benchmarks)
- **Level 0 (Floor):** 3-gram and 5-gram with Kneser-Ney smoothing on the exact tokenized split.
- **Level 1 (Standard Baselines):**
  1. **GRU / LSTM:** 2–3 layers.
  2. **Decoder-only Transformer:** 4–6 layers, full causal attention.
  3. **Sliding-Window Attention Transformer:** Attention window restricted to $W=128$ (matching the NCA's receptive field). This prevents confounding architectural bias with raw context span.
  4. **Mamba / RWKV:** Reference linear SSM.

---

## 5. Question 2 — What Does Its Computation Look Like? (Level 2 Core Architecture)

To test whether the computational driver is **cellular iteration** or merely a **deep convolutional stack**, the architecture explicitly implements and compares:
1. **Shared NCA Rule ($F_\theta$):** The exact same convolution weights and GRU gates are reused across all $K$ microsteps; only the dilation $d_k$ changes per step.
2. **Step-Specific CNN ($F_{\theta_k}$):** Independent weights per step (a standard causal dilated ConvNet baseline).

### 5.1 The Causal NCAStep Specification

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalNCAStep(nn.Module):
    """
    Implements a causal, dilated local update step.
    Supports both True NCA (Shared Rule) and Stacked CNN (Step-Specific) modes.
    """
    def __init__(self, d_model, radius=2, d_hidden=None, max_K=8, shared_weights=True):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden or (d_model * 2)
        self.radius = radius
        self.kernel_size = radius + 1
        self.max_K = max_K
        self.shared_weights = shared_weights
        
        # Exponential dilation schedule: [1, 2, 4, 8, 16, 32, ...]
        self.dilations = [2**i for i in range(max_K)]

        if self.shared_weights:
            # TRUE NCA RULE: Single weight tensor reused across all steps K
            self.conv_weight = nn.Parameter(
                torch.randn(self.d_hidden, d_model, self.kernel_size) * 
                (2.0 / (d_model * self.kernel_size))**0.5
            )
            self.conv_bias = nn.Parameter(torch.zeros(self.d_hidden))
        else:
            # UNTIED BASELINE: Step-specific convolution weights
            self.convs = nn.ModuleList([
                nn.Conv1d(d_model, self.d_hidden, kernel_size=self.kernel_size, padding=0)
                for _ in range(max_K)
            ])

        # Step embedding provides micro-step awareness
        self.step_embed = nn.Parameter(torch.randn(max_K, self.d_hidden, 1) * 0.02)

        # GRU gating mechanism (1x1 convs over channel dimension)
        self.update_gate = nn.Conv1d(self.d_hidden + d_model, d_model, kernel_size=1)
        self.reset_gate = nn.Conv1d(self.d_hidden + d_model, d_model, kernel_size=1)
        self.candidate_nhood = nn.Conv1d(self.d_hidden, d_model, kernel_size=1)
        self.candidate_state = nn.Conv1d(d_model, d_model, kernel_size=1)

    def forward(self, s, step_idx):
        d = self.dilations[step_idx]
        pad_len = self.radius * d
        
        # STRICT LEFT PAD ONLY — guarantees zero future information leakage
        s_padded = F.pad(s, (pad_len, 0))
        
        if self.shared_weights:
            # Apply the SAME learned rule F_theta with step-specific dilation d
            neighborhood = F.conv1d(s_padded, self.conv_weight, self.conv_bias, dilation=d)
        else:
            neighborhood = F.conv1d(s_padded, self.convs[step_idx].weight, self.convs[step_idx].bias, dilation=d)

        neighborhood = neighborhood + self.step_embed[step_idx]
        neighborhood = F.silu(neighborhood)

        # Correct GRU recurrence across microsteps
        joint = torch.cat([neighborhood, s], dim=1)
        z = torch.sigmoid(self.update_gate(joint))
        r = torch.sigmoid(self.reset_gate(joint))
        
        cand = torch.tanh(self.candidate_nhood(neighborhood) + self.candidate_state(r * s))
        s_new = (1.0 - z) * s + z * cand
        return s_new
```

### 5.2 Full NCA-LM Model

```python
class NCA_LM(nn.Module):
    def __init__(self, vocab_size, d_embed=256, d_hidden_channels=256, radius=2, K=6, shared_weights=True):
        super().__init__()
        self.d_embed = d_embed
        self.d_model = d_embed + d_hidden_channels
        self.K = K
        self.embed = nn.Embedding(vocab_size, d_embed)
        self.step = CausalNCAStep(
            self.d_model, radius=radius, max_K=K, shared_weights=shared_weights
        )
        self.readout = nn.Linear(self.d_model, vocab_size)
        if self.d_model == d_embed:
            self.readout.weight = self.embed.weight

    def forward(self, x, override_K=None):
        B, T = x.shape
        e = self.embed(x).transpose(1, 2)  # [B, d_embed, T]
        
        h0_dim = self.d_model - self.d_embed
        if h0_dim > 0:
            h0 = torch.zeros(B, h0_dim, T, device=x.device, dtype=e.dtype)
            s = torch.cat([e, h0], dim=1)
        else:
            s = e
            
        steps = override_K if override_K is not None else self.K
        for k in range(steps):
            s = self.step(s, step_idx=k)
            
        logits = self.readout(s.transpose(1, 2))
        return logits
```

### 5.3 The Shared vs. Unshared Experiment
Train both variants on WikiText-2 with matched parameter budgets (compensating width as needed):
- **Hypothesis:** Does reusing the *exact same rule* $F_\theta$ across dilated steps match or approximate the expressivity of the unshared stack $F_{\theta_k}$?
- If yes: cellular iteration is doing real computational work.
- If no: parameter count was the confounding factor, exactly like in KdV.

---

## 6. Question 3 — Does NCA Exhibit an Unusual / Unpredicted Capability?

Rather than assuming what NCA is good at, we test five distinct candidate behaviors:

### Candidate A: Test-Time Computation-Depth Scaling
$$\boxed{ \text{Train at } K=4 \implies \text{Evaluate at } K \in \{1, 2, 4, 6, 8\} }$$
- Does increasing micro-step iterations at inference time improve perplexity on tokens with high next-token entropy?
- Does more computation depth allow the model to "think longer" about ambiguous context?
- Measure quality vs. FLOPs trade-off curve.

### Candidate B: Independent Decoupling of Receptive Field, Computation, and Context
Do not just evaluate length generalization out to 1,024 and claim victory from a flat slope. Instead, independently vary:
1. **Receptive Field ($RF$):** Vary $r$ and dilation schedule at fixed $K$.
2. **Computation ($K$):** Vary micro-steps at fixed $RF$.
3. **Context Length ($T$):** Evaluate at $T \in \{128, 256, 512, 1024\}$.
- **The Honest Comparison:** At the **same accessible context window** (e.g., $RF = 128$), compare NCA-LM against the Sliding-Window Attention Transformer on both absolute PPL and FLOP efficiency.

### Candidate C: Perturbation Recovery & Representational Homeostasis
Directly inherited from the KdV memory disruption methodology:
1. Stream autoregressive text generation for 500 tokens.
2. At token 250, inject noise into the internal activation buffer (zeroing, Gaussian noise, or token shuffling).
3. **Metric:** Measure recovery trajectory — how many tokens does it take for next-token cross-entropy to return within 5% of undisturbed baseline?
4. Compare against Mamba (latent state perturbation) and Transformer (KV cache perturbation).

### Candidate D: Representational Robustness Under Surface Noise
- Test resistance to character-level noise (typos, insertions, deletions) and subword masking.
- Measure perplexity degradation slope as corruption probability $p \in [0.0, 0.2]$ increases.

### Candidate E: Activation Memory Footprint in Streaming
- Benchmark empirical inference memory vs. stream length $T \in [100, 10000]$.
- Confirm $O(1)$ activation buffer scaling of causal dilated conv vs. $O(T)$ KV-cache growth.

---

## 7. Question 4 — Can That Capability Improve a Conventional Model?

$$\boxed{ \text{NCA as a Component} > \text{NCA as a Transformer Replacement} }$$

If Question 3 reveals a distinct capability (e.g., test-time compute depth, noise smoothing, or homeostasis), we do **not** force NCA to replace the entire Transformer. Instead, we insert it as an **adaptor component**:

```
Input Tokens → Embedding → Transformer Layer 1 → NCA Adaptor Step (K=2) → Transformer Layer 2 → ... → Readout
```

- Target parameter overhead: $<5\%$.
- Test whether the hybrid model inherits the specific capability identified in Question 3 (e.g., typo tolerance, test-time compute scaling) without compromising the Transformer's global attention quality.

---

## 8. Decision Gates (Stop/Go Protocol)

- **Gate 1 (Q1 — Viability):** Does NCA-LM establish a functioning next-token predictor substantially above the n-gram baseline, and does it exhibit meaningful learning dynamics as $K$ and receptive field vary? If it cannot outperform the n-gram floor or shows flat learning across $K$/receptive field variations, stop and diagnose before investing further compute.
- **Gate 2 (Q2 — Shared vs. Unshared):** Does the shared-rule NCA ($F_\theta$) show meaningful learning dynamics compared to the unshared stack ($F_{\theta_k}$)? Quantify the weight-sharing penalty.
- **Gate 3 (Q3 — Candidate Search):** Does at least **one** candidate behavior (A: compute depth, B: context decoupling, C: perturbation recovery, D: robustness, E: streaming) demonstrate a measurable, reproducible advantage over the sliding-window Transformer and Mamba?
  - *If yes:* Proceed directly to Question 4 and design the Hybrid Adaptor around that specific behavior.
  - *If no:* Halt. Conclude with a rigorous negative result detailing which properties failed to transfer from 2D vision/physics to 1D language.
- **Gate 4 (Q4 — Hybrid Transfer):** Does the NCA-adaptor deliver a statistically significant, reproducible advantage on the target capability at $<5\%$ parameter overhead? If so, draft the paper.

---

## 9. Compute Budget & Timeline (~24GB GPU)

| Phase | Duration | Focus |
|---|---|---|
| **Phase 1 (Q1)** | 3–4 days | Tokenizer, data prep, n-gram floor, Transformer & Mamba baselines |
| **Phase 2 (Q2)** | 5–7 days | Shared vs. Unshared NCA-LM implementation and calibration |
| **Phase 3 (Q3)** | 1.5–2 weeks | Probing the 5 candidate behaviors (depth scaling, recovery, robustness) |
| **Phase 4 (Q4)** | 1–2 weeks | Hybrid component integration & evaluation (contingent on Q3) |
| **Total** | **~4–5 weeks** | Structured, disciplined empirical investigation |
