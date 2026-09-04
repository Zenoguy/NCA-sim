# NCA-LM Implementation Plan
### What can NCA bring to generative modeling / deep learning that conventional architectures don't naturally provide?

Author's context: continuation of the KdV-NCA project. This plan operationalizes the experimental ladder (Levels 0–5) into a concrete, buildable research codebase, with dataset choices, architecture specs, hyperparameters, compute budgets, and — critically — stop/go gates so this doesn't turn into a second KdV-style detour.

---

## 0. Repositioning, based on current literature (read this before writing code)

Four corrections to make before implementation starts, because they change what you build:

1. **The "O(1) memory streaming" claim is not free territory.** State-space models (Mamba/Mamba-2/Mamba-3) and RWKV already own this niche as their entire value proposition — linear-time, constant-memory sequence modeling with transformer-competitive quality, actively developed through 2026 (Mamba-3 adds complex-valued state updates). If your headline claim for Level 4 is "NCA has bounded memory unlike Transformers," reviewers will immediately ask "so does Mamba, and better." **You need a baseline that includes a small Mamba/RWKV model, not just Transformer vs NCA.**

2. **Your actual differentiator vs. SSMs is per-token iterative depth, not memory boundedness.** Mamba/RWKV do *one* state update per token. An NCA-LM does *K* local update steps over the sequence representation before emitting a prediction — that connects to the recurrent-depth / looped-transformer test-time-compute literature (models that iterate a shared block K times at inference to trade compute for quality), which is a hot, distinct research thread from SSMs. **Reframe the contribution as "adjustable per-step computation depth with bounded local communication," positioned against both Transformers (no depth-compute tradeoff at inference) and SSMs (no per-token iteration at all)**, not just "has memory, Transformer doesn't."

3. **Causal masking is a first-class implementation detail, not an afterthought.** A naive local NCA update (symmetric neighbor convolution across the full sequence, repeated K times) will leak information from position i+1, i+2, ... backward into the state at position i after just 1 step. This invalidates next-token prediction — your model would be trivially "solving" the task by peeking at the answer. Every NCA-LM update must be **strictly causal at every micro-step**, not just causal in the final readout. This is spelled out in §6.

4. **Receptive field scaling is non-negotiable for language (The FIR Receptive Field Trap).** An NCA with static radius $r=4$ repeated $K=4$ times has an effective receptive field of only $K \times r = 16$ tokens (~2–3 words). On WikiText-2, a 16-token model will produce catastrophic perplexity (PPL > 150) simply because it cannot see sentence-level context. *TextNCA* (Aug 2026) solved this by using **hierarchical local attention with expanding windows** ($w \in \{8, 32, 128\}$). For convolutional NCA, you **must use causal dilated convolutions** ($d_k = 2^k$) across the micro-steps so that $K=6$ reaches an effective receptive field of $\ge 126$ tokens without exploding parameter counts.

---

## 1. Success criteria (write these down now, check against them at each gate)

The project succeeds if it produces **at least one** of:
- **Level 3 Hybrid Robustness (Highest Priority & Primary Value Proposition):** A quantified regime where an NCA-adaptor hybrid improves a Transformer's robustness (character-level typos, subword drops, OOD text) at <5% parameter overhead, translating AdaNCA's (NeurIPS 2024) vision findings to language.
- **Length Generalization (with honest controls):** A quantified regime where NCA-LM's degradation curve (perplexity vs. extrapolated sequence length) is measurably flatter than a matched Transformer, evaluated with **both** absolute perplexity reported and a **sliding-window attention Transformer** included as an explicit context-matched baseline (§7).
- **Streaming & Compute-Depth Trade-off:** A quantified inference-depth advantage where increasing micro-step iterations $K$ at inference systematically recovers perplexity on harder/perturbed sequences, compared against fixed-depth SSMs (Mamba) and KV-cached Transformers.

It fails informatively (still worth writing up) if none of these hold **and** you can point to which specific axis (locality radius, gating, iteration count, staging schedule) was ruled out and why — same standard as your KdV memory-gating negative result.

---

## 2. Repo structure

```
nca-lm/
├── configs/                    # one YAML per experiment, hydra or plain argparse
│   ├── level0_ngram.yaml
│   ├── level1_gru.yaml
│   ├── level1_transformer.yaml
│   ├── level1_transformer_sliding.yaml # context-matched sliding window control
│   ├── level1_mamba.yaml        # small SSM baseline, added per §0
│   ├── level2_nca.yaml
│   ├── level3_hybrid.yaml
│   └── pe_ablation/*.yaml       # positional-encoding / length-generalization matrix
├── data/
│   ├── prepare_wikitext2.py
│   ├── prepare_wikitext103.py
│   └── tokenizer.py             # byte-level BPE, small vocab (~8k-16k)
├── models/
│   ├── ngram.py
│   ├── rnn_baseline.py          # GRU/LSTM
│   ├── transformer_baseline.py  # decoder-only, swappable PE + sliding-window mode
│   ├── mamba_baseline.py        # thin wrapper around an existing Mamba impl
│   ├── nca_lm.py                # the core contribution (dilated causal NCA)
│   └── hybrid.py                # Transformer block + NCA adaptor
├── train.py                     # single entrypoint, config-driven
├── eval/
│   ├── perplexity.py
│   ├── length_generalization.py # train-short/test-long harness with absolute & relative metrics
│   ├── streaming_bench.py       # latency/memory vs. context length
│   └── noise_robustness.py      # char-level corruption & typo injection for hybrid eval
├── notebooks/
│   └── param_matching.ipynb     # verify matched parameter counts across models
└── README.md                    # decision log — append, never rewrite history
```

Use one shared `train.py` and `Dataset`/`Tokenizer` across every model so comparisons are apples-to-apples. This is the single most common way these comparisons go wrong — different tokenizers or context windows per model silently invalidate the whole ladder.

---

## 3. Shared infrastructure

**Datasets** (pick based on iteration speed, not final ambition):
- **WikiText-2** (~2M tokens) — fast iteration, architecture debugging, param-matching sanity checks. Train an epoch in minutes on a single consumer GPU.
- **WikiText-103** (~100M tokens) — the scale TextNCA used; needed if you want results directly comparable to that paper.
- **enwik8** (byte-level, 100MB) — optional, useful if you want to sidestep tokenizer choice entirely and test raw character/byte sequence modeling.

Start on WikiText-2 for every architecture/debugging pass. Only move a config to WikiText-103 once it trains stably and matches expected loss curves at small scale.

**Tokenizer:** byte-level BPE, vocab size 8k–16k, trained once, shared across all models and configs. Do not let different architectures use different tokenizers — this is a silent confound.

**Parameter matching:** before running any comparison, write a small script that instantiates every model at the target parameter count (e.g., 10M ± 5%) and asserts it. Log actual param counts in every run's metadata. Your KdV numbers already showed how much variance exists between "matched" configs — don't let a 20% param mismatch masquerade as an architectural finding.

**Sequence length / context:** fix a base training length (e.g., 256 tokens) shared across Levels 0–3. Length-generalization experiments (§7) extend the *eval* length only, never retrain at a different length within the same comparison.

---

## 4. Level 0 — n-gram baseline

Trivial but do it anyway — it's the "did the pipeline even work" sanity check and gives you a real floor.

- 3-gram and 5-gram with Kneser-Ney smoothing (KenLM or a simple Python implementation on the tokenized corpus).
- Report perplexity on the same held-out split every neural model will use.
- **Time budget: half a day.** If this takes longer, something's wrong with the data pipeline, not the n-gram model.

---

## 5. Level 1 — conventional neural baselines

Four models, all parameter-matched to your target NCA-LM size (start with ~10M params):

- **GRU/LSTM** — 2–3 layers, standard next-token loss.
- **Standard decoder-only Transformer** — 4–6 layers, matched d_model/heads to hit the param target. Full causal attention.
- **Sliding-Window Attention Transformer (Critical Control)** — Same architecture, but attention window restricted to match NCA-LM's maximum receptive field (e.g., $W=128$). This isolates the effect of local inductive bias from mere context window size.
- **Small Mamba or RWKV baseline** — per §0, do not skip this. Use an existing open implementation (`mamba-ssm` package, or a minimal RWKV reference implementation) rather than reimplementing from scratch; you need it as a reference point, not as a research contribution in itself.

Each Transformer variant gets a PE-swap variant per §7 (absolute PE, RoPE, and NoPE).

**Time budget: 3–5 days**, mostly infra/debugging, not tuning. Expected result: Full Transformer $\ge$ Mamba $\ge$ Sliding-Window Transformer $\ge$ GRU at this scale. This is calibration, not a finding — don't over-tune.

---

## 6. Level 2 — NCA-LM architecture (the core build)

### 6.1 State representation

For a sequence of length T, position i has state:
```
s_i^0 = [ e(x_{i-1}), h_i^0 ]        # shifted input, standard autoregressive convention
```
where `e(x_{i-1})` is the token embedding of the *previous* token (standard GPT-style shift — position i predicts x_i using only x_{<i}), and `h_i^0` is a learned or zero-initialized hidden channel of dimension `d_h`. Total per-cell state dimension `d_model = d_embed + d_h`.

### 6.2 Causal Dilated Local Update Rule (Corrected PyTorch Spec)

To solve the 16-token receptive field bottleneck while strictly maintaining causality, we use **exponentially dilated causal convolutions** across micro-steps ($d_k = 2^k$). With $radius=2$ ($k=3$) and $K=6$, the receptive field expands to:
$$\text{Receptive Field} = \text{radius} \times \sum_{k=0}^{K-1} 2^k = 2 \times (1 + 2 + 4 + 8 + 16 + 32) = 126 \text{ tokens.}$$

Furthermore, we fix the dead `reset_gate` parameter and dimension mismatch bugs:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalNCAStep(nn.Module):
    def __init__(self, d_model, radius=2, d_hidden=None, max_K=8):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden or (d_model * 2)
        self.radius = radius
        self.kernel_size = radius + 1
        self.max_K = max_K

        # Exponential dilation schedule: [1, 2, 4, 8, 16, 32, ...]
        self.dilations = [2**i for i in range(max_K)]
        
        # Bank of causal convs with shared channel dimensions but step-specific dilations
        self.causal_convs = nn.ModuleList([
            nn.Conv1d(
                d_model, 
                self.d_hidden, 
                kernel_size=self.kernel_size, 
                dilation=d, 
                padding=0
            )
            for d in self.dilations
        ])

        # Step embedding correctly matches d_hidden
        self.step_embed = nn.Parameter(torch.randn(max_K, self.d_hidden, 1) * 0.02)

        # 1x1 Convolutions for GRU gating directly over [B, C, T] tensors
        # Correct GRU recurrence: reset gate gates previous state s before candidate projection
        self.update_gate = nn.Conv1d(self.d_hidden + d_model, d_model, kernel_size=1)
        self.reset_gate = nn.Conv1d(self.d_hidden + d_model, d_model, kernel_size=1)
        self.candidate_nhood = nn.Conv1d(self.d_hidden, d_model, kernel_size=1)
        self.candidate_state = nn.Conv1d(d_model, d_model, kernel_size=1)

    def forward(self, s, step_idx):
        """
        Args:
            s: [B, d_model, T] state tensor
            step_idx: int index from 0 to K-1
        Returns:
            s_new: [B, d_model, T] updated state
        """
        d = self.dilations[step_idx]
        pad_len = self.radius * d
        
        # STRICT LEFT PAD ONLY — enforces causality at every micro-step
        s_padded = F.pad(s, (pad_len, 0))
        neighborhood = self.causal_convs[step_idx](s_padded)  # [B, d_hidden, T]
        neighborhood = neighborhood + self.step_embed[step_idx]
        neighborhood = F.silu(neighborhood)

        # GRU-style gating
        joint = torch.cat([neighborhood, s], dim=1)  # [B, d_hidden + d_model, T]
        z = torch.sigmoid(self.update_gate(joint))     # Update gate [B, d_model, T]
        r = torch.sigmoid(self.reset_gate(joint))      # Reset gate [B, d_model, T]

        # r correctly modulates the previous state s
        cand = torch.tanh(self.candidate_nhood(neighborhood) + self.candidate_state(r * s))
        s_new = (1.0 - z) * s + z * cand
        return s_new
```

### 6.3 Full Model

```python
class NCA_LM(nn.Module):
    def __init__(self, vocab_size, d_embed=256, d_hidden_channels=256, radius=2, K=6, tie_weights=True):
        super().__init__()
        self.d_embed = d_embed
        self.d_model = d_embed + d_hidden_channels
        self.K = K
        self.embed = nn.Embedding(vocab_size, d_embed)
        self.step = CausalNCAStep(self.d_model, radius=radius, max_K=K)
        
        self.readout = nn.Linear(self.d_model, vocab_size)
        if tie_weights and (self.d_model == d_embed):
            self.readout.weight = self.embed.weight
        elif tie_weights:
            self.readout_proj = nn.Linear(self.d_model, d_embed, bias=False)
            self.readout.weight = self.embed.weight

    def forward(self, x):
        # x: [B, T] token ids (input = x[:, :-1], target = x[:, 1:])
        B, T = x.shape
        e = self.embed(x).transpose(1, 2)  # [B, d_embed, T]
        
        # Initialize persistent hidden channels h0 with zeros or learned state
        h0_dim = self.d_model - self.d_embed
        if h0_dim > 0:
            h0 = torch.zeros(B, h0_dim, T, device=x.device, dtype=e.dtype)
            s = torch.cat([e, h0], dim=1)  # [B, d_model, T]
        else:
            s = e
            
        # Parallel micro-step updates across sequence
        for k in range(self.K):
            s = self.step(s, step_idx=k)
            
        if hasattr(self, 'readout_proj'):
            s_out = self.readout_proj(s.transpose(1, 2))  # [B, T, d_embed]
            logits = F.linear(s_out, self.embed.weight)
        else:
            logits = self.readout(s.transpose(1, 2))      # [B, T, vocab]
            
        return logits
```

This preserves **parallel training** — every micro-step is a batched 1D causal convolution across the sequence, keeping $O(T)$ training parallelism.

### 6.4 Hyperparameter Sweep (Level 2 Core Experiment)

With dilation enabled, $K$ now scales receptive field exponentially:

| Axis | Values to test | Receptive Field (r=2) | Expected Outcome |
|---|---|---|---|
| K (depth) | 1, 2, 4, 6, 8 | 2, 6, 30, 126, 510 tokens | Rapid PPL gain up to K=6; plateau/diminishing returns beyond K=6 |
| Dilation | Exponential ($2^k$) vs. Linear ($k$) vs. None ($1$) | 126 vs. 42 vs. 12 (at K=6) | Exponential dilation drastically outperforms no-dilation |
| Gating | Full GRU vs. Update-only (Highway) vs. Additive | Fixed | Full GRU required for stable multi-step propagation |

**Expected Headline Result:** Standalone NCA-LM will perform competitively with the *Sliding-Window Transformer* ($W=128$), but will still lag behind the full-context Transformer and Mamba. **Do not burn compute trying to close the gap on full-context WikiText.** Move immediately to §7 and Level 3.

---

## 7. Positional-Encoding & Length Generalization (The "No Vanity Metric" Protocol)

> [!WARNING]
> **The Degenerate Flat-Curve Trap:** A local convolutional model with receptive field $W$ trivially exhibits 0% degradation when evaluated at length 1024 vs 128 because position $i$ mathematically cannot read beyond $i-W$. Claiming "superior length generalization" because the curve is flat while absolute perplexity is mediocre is a hollow vanity metric.

### Corrected Protocol:
1. **Models to Compare:**
   - Standard Transformer (with Absolute PE)
   - Standard Transformer (NoPE / "NoPos", Haviv et al. / Kazemnejad et al.)
   - Standard Transformer (with RoPE)
   - **Sliding-Window Transformer ($W=128$, RoPE) [Mandatory Control]**
   - NCA-LM (Receptive Field = 126)
   - Mamba / RWKV (recurrent reference)
2. **Lengths:** Train all models at length $T=128$. Evaluate at $T \in \{128, 256, 512, 1024\}$.
3. **Required Dual Reporting:**
   - Plot **Absolute Perplexity vs. Test Length** on the primary axis.
   - Plot **Relative Degradation Ratio** ($\text{PPL}_{1024} / \text{PPL}_{128}$) on the secondary axis.
   - **Scientific hypothesis:** If NCA-LM matches or beats the *Sliding-Window Transformer* on both absolute PPL and stability beyond $T=128$, the structural relative inductive bias of NCA is empirically validated.

---

## 8. Level 3 — Hybrid: NCA as Robustness Adaptor (The Primary Research Opportunity)

This is the **strongest, most publishable hypothesis of the project**, translating AdaNCA (NeurIPS 2024) from vision to language:

```
Input Tokens → Embedding → Transformer Block 1 → NCA Adaptor (K=2) → Transformer Block 2 → ... → Output Logits
```

### 8.1 Why This Works Conceptually
Vision Transformers lack local spatial inductive bias; AdaNCA showed that inserting lightweight local NCA adaptors between ViT layers improved adversarial robustness by +10% at <3% parameter cost. 

In NLP, tokenized representations suffer extreme brittleness to surface perturbations: a single typo or OCR error maps to a wildly different subword token, disrupting global self-attention. A local NCA adaptor applies **iterative local diffusion and error-correction** in latent space, restoring representational stability.

### 8.2 Evaluation Protocol
1. **Parameter Overhead:** Insert 2-step NCA adaptors between 2 intermediate Transformer layers; constrain overhead to $<5\%$ total model parameters.
2. **Clean Perplexity:** Confirm clean WikiText-2 perplexity does not degrade significantly ($< 0.5$ PPL delta).
3. **Robustness Benchmark (The Key Test):** Inject synthetic corruptions into the test set:
   - Character swaps, insertions, deletions (simulating typos).
   - Subword masking / dropout (simulating missing text).
   - Evaluate degradation curve: $\Delta \text{Perplexity}$ as corruption probability $p \in [0.0, 0.2]$ increases.
   - **Hypothesis:** Transformer + NCA Adaptor retains substantially lower perplexity under noise than vanilla Transformer.

**Time budget: 1.5–2 weeks.** Focus the bulk of the project's analytical depth here.

---

## 9. Level 4 — Streaming Inference & State Persistence

Only run this if Level 2/3 demonstrated compelling local dynamics.

### 9.1 Mathematical Reality of Streaming Dilated Convolutions
Unlike RNNs or SSMs (which are IIR dynamical systems with latent state vectors), an FIR causal convolution's state buffer is a **FIFO activation cache**.
- At each token generation step, the model only stores past activations of length $d_k \times \text{radius}$ for each layer $k$.
- Total memory per token is $O(K \cdot \text{radius} \cdot d_{max})$, strictly constant $O(1)$ with respect to sequence length $T$.
- Compare against KV-cached Transformer ($O(T)$ memory growth) and Mamba ($O(1)$ latent state).

### 9.2 The State Disruption Probe
Reusing the KdV state-swapping methodology:
1. Stream 500 tokens of text.
2. At token 250, perturb the layer cache (zeroing, Gaussian noise, or swapping with another sequence).
3. Measure recovery time (how many tokens until next-token cross-entropy returns to baseline).
4. Contrast against Mamba's latent state disruption.

---

## 10. Level 5 — Conditional Architectural Replacement

Explicitly gated: only attempt if Levels 2–4 produced an unambiguous, reproducible advantage on a specific capability (e.g., local noise filtering or streaming efficiency). Otherwise, conclude with the Level 3 Hybrid paper.

---

## 11. Compute Budget (Single GPU, ~24GB class card)

| Phase | Wall-clock | Focus |
|---|---|---|
| Infra + Level 0 | 1–2 days | Tokenizer, data loaders, sanity check |
| Level 1 (Baselines) | 3–4 days | Transformer, Sliding Transformer, Mamba, GRU |
| §7 Length Generalization | 2–3 days | Eval-only passes across sequence lengths |
| Level 2 (Dilated NCA-LM) | 5–7 days | K-sweep, dilation verification |
| Level 3 (Hybrid Adaptor & Noise) | 1.5–2 weeks | Primary paper results: clean PPL + corruption robustness |
| Level 4 (Streaming, conditional) | 1 week | Activation caching benchmark & recovery probes |
| **Total** | **~4 weeks** | Highly scoped, publication-ready trajectory |

---

## 12. Decision Gates (Hard Stops)

1. **Gate 1 (Post-Level 2 Sweep):** If dilated NCA-LM ($K=6$) fails to match the *Sliding-Window Transformer* ($W=128$) within reasonable bounds on WikiText-2, stop and diagnose before running further sweeps.
2. **Gate 2 (Post-§7 Length Test):** If NCA-LM's absolute perplexity is non-competitive despite a flat relative slope, report it transparently as an architectural limitation of compact receptive fields, not a length-generalization victory.
3. **Gate 3 (Post-Level 3 Robustness):** If the NCA adaptor does not improve text corruption robustness over the baseline Transformer, halt the project and write up the negative result. If it does improve robustness by $>15\%$ under noise, prioritize writing the paper on **"AdaNCA-Text: Robust Language Representations via Cellular Adaptors."**

---

## 13. Logging & Scientific Reproducibility

- Every run logs: config YAML, exact git commit, parameter count assertion, seed, training FLOPs, and evaluation metrics.
- Keep the append-only `README.md` log active throughout all phases.
