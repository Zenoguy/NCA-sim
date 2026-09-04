# NCA-LM Implementation Plan
### What can NCA bring to generative modeling / deep learning that conventional architectures don't naturally provide?

Author's context: continuation of the KdV-NCA project. This plan operationalizes the experimental ladder (Levels 0–5) into a concrete, buildable research codebase, with dataset choices, architecture specs, hyperparameters, compute budgets, and — critically — stop/go gates so this doesn't turn into a second KdV-style detour.

---

## 0. Repositioning, based on current literature (read this before writing code)

Three corrections to make before implementation starts, because they change what you build:

1. **The "O(1) memory streaming" claim is not free territory.** State-space models (Mamba/Mamba-2/Mamba-3) and RWKV already own this niche as their entire value proposition — linear-time, constant-memory sequence modeling with transformer-competitive quality, actively developed through 2026 (Mamba-3 adds complex-valued state updates). If your headline claim for Level 4 is "NCA has bounded memory unlike Transformers," reviewers will immediately ask "so does Mamba, and better." **You need a baseline that includes a small Mamba/RWKV model, not just Transformer vs NCA.**

2. **Your actual differentiator vs. SSMs is per-token iterative depth, not memory boundedness.** Mamba/RWKV do *one* state update per token. Your NCA-LM does *K* local update steps per token position before emitting a prediction — that's an adjustable "thinking per token" axis that SSMs don't have. This connects to the recurrent-depth / looped-transformer test-time-compute literature (models that iterate a shared block K times at inference to trade compute for quality), which is a hot, distinct research thread from SSMs. **Reframe Level 4's contribution as "adjustable per-step computation depth with bounded state," positioned against both Transformers (no depth-compute tradeoff at inference) and SSMs (no per-token iteration at all)**, not just "has memory, Transformer doesn't."

3. **Causal masking is a first-class implementation detail, not an afterthought.** A naive local NCA update (symmetric neighbor convolution across the full sequence, repeated K times) will leak information from position i+1, i+2, ... backward into the state at position i after just 1 step. This invalidates next-token prediction — your model would be trivially "solving" the task by peeking at the answer. Every NCA-LM update must be **strictly causal at every micro-step**, not just causal in the final readout. This is spelled out in §6.

---

## 1. Success criteria (write these down now, check against them at each gate)

The project succeeds if it produces **at least one** of:
- A quantified regime where NCA-LM's degradation curve (perplexity vs. extrapolated sequence length) is measurably flatter than a matched Transformer's, with an explanation grounded in the causal-mask/positional-information literature (§7).
- A quantified regime where an NCA-adaptor hybrid (Level 3) improves a Transformer's robustness (character-level noise, OOD text) at <5% parameter overhead, mirroring AdaNCA's finding for vision but for text.
- A quantified streaming-inference advantage (latency or memory vs. context length) for NCA-LM against both a Transformer-with-KV-cache **and** a small Mamba/RWKV baseline — not just against Transformer.

It fails informatively (still worth writing up) if none of these hold **and** you can point to which specific axis (locality radius, gating, iteration count, staging schedule) was ruled out and why — same standard as your KdV memory-gating negative result.

---

## 2. Repo structure

```
nca-lm/
├── configs/                    # one YAML per experiment, hydra or plain argparse
│   ├── level0_ngram.yaml
│   ├── level1_gru.yaml
│   ├── level1_transformer.yaml
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
│   ├── transformer_baseline.py  # decoder-only, swappable PE
│   ├── mamba_baseline.py        # thin wrapper around an existing Mamba impl
│   ├── nca_lm.py                # the core contribution
│   └── hybrid.py                # Transformer block + NCA adaptor
├── train.py                     # single entrypoint, config-driven
├── eval/
│   ├── perplexity.py
│   ├── length_generalization.py # train-short/test-long harness
│   ├── streaming_bench.py       # latency/memory vs. context length
│   └── noise_robustness.py      # char-level corruption for hybrid eval
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
- **enwik8** (byte-level, 100MB) — optional, useful if you want to sidestep tokenizer choice entirely and test raw sequence modeling.

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

Three models, all parameter-matched to your target NCA-LM size (start with ~10M params):

- **GRU/LSTM** — 2–3 layers, standard next-token loss.
- **Small decoder-only Transformer** — 4–6 layers, matched d_model/heads to hit the param target. This is your main comparison point.
- **Small Mamba or RWKV baseline** — per §0, do not skip this. Use an existing open implementation (`mamba-ssm` package, or a minimal RWKV reference implementation) rather than reimplementing from scratch; you need it as a reference point, not as a research contribution in itself.

Each gets a PE-swap variant per §7 (absolute PE / no PE) for the Transformer specifically — the RNN and Mamba baselines don't need this since recurrence gives them order for free.

**Time budget: 3–5 days**, mostly infra/debugging, not tuning. Expected result (per TextNCA/CellARC precedent): Transformer ≥ Mamba/RWKV ≥ GRU at this scale. This is calibration, not a finding — don't over-tune.

---

## 6. Level 2 — NCA-LM architecture (the core build)

### 6.1 State representation

For a sequence of length T, position i has state:
```
s_i^0 = [ e(x_{i-1}), h_i^0 ]        # shifted input, standard autoregressive convention
```
where `e(x_{i-1})` is the token embedding of the *previous* token (standard GPT-style shift — position i predicts x_i using only x_{<i}), and `h_i^0` is a learned or zero-initialized hidden channel of dimension `d_h`. Total per-cell state dimension `d = d_e + d_h`.

### 6.2 Causal local update rule (critical — see §0.3)

```python
class CausalNCAStep(nn.Module):
    def __init__(self, d_model, radius=4, d_hidden=None):
        super().__init__()
        d_hidden = d_hidden or d_model * 2
        k = radius + 1  # kernel sees [i-radius, ..., i], never i+1..
        self.causal_conv = nn.Conv1d(
            d_model, d_hidden, kernel_size=k, padding=0
        )
        self.radius = radius
        # GRU-style gate — TextNCA found this necessary for iteration to help
        self.update_gate = nn.Linear(d_hidden, d_model)
        self.reset_gate = nn.Linear(d_hidden, d_model)
        self.candidate = nn.Linear(d_hidden, d_model)
        self.step_embed = None  # set per-K, see 6.3

    def forward(self, s, step_idx):
        # s: [B, d_model, T]
        s_padded = F.pad(s, (self.radius, 0))  # LEFT pad only — enforces causality
        neighborhood = self.causal_conv(s_padded)  # [B, d_hidden, T]
        neighborhood = neighborhood + self.step_embed[step_idx]  # learned per-step embedding
        neighborhood = F.silu(neighborhood)

        z = torch.sigmoid(self.update_gate(neighborhood.transpose(1,2)))
        r = torch.sigmoid(self.reset_gate(neighborhood.transpose(1,2)))
        cand = torch.tanh(self.candidate(neighborhood.transpose(1,2)))
        s_new = (1 - z) * s.transpose(1,2) + z * cand
        return s_new.transpose(1,2)
```

Key details, each tied directly to a finding from your prior research or the literature review:
- **Left-only padding on the causal conv** is what prevents future-token leakage. Verify this with a unit test: zero out all tokens at position > i, confirm logits at position i are unchanged. This is the single most important correctness check in the whole project — run it before trusting any perplexity number.
- **Per-step learned embeddings** (`step_embed[step_idx]`) — TextNCA found these necessary (along with the GRU gate) for the iteration-depth axis to produce any benefit at all. Don't skip this to save code; it's specifically flagged as required, not optional, by the closest prior work.
- **Radius (neighbor window) and depth K jointly determine receptive field**, causally: `receptive_field = radius * K`. Log this number in every config — it's your controlled variable when comparing against Transformer's full-context or Mamba's full-history compression.

### 6.3 Full model

```python
class NCA_LM(nn.Module):
    def __init__(self, vocab_size, d_embed, d_hidden_channels, radius, K, tie_weights=True):
        super().__init__()
        d_model = d_embed + d_hidden_channels
        self.embed = nn.Embedding(vocab_size, d_embed)
        self.K = K
        self.step = CausalNCAStep(d_model, radius=radius)
        self.step.step_embed = nn.Parameter(torch.randn(K, d_model, 1) * 0.02)
        self.readout = nn.Linear(d_model, vocab_size)
        if tie_weights:
            self.readout.weight = self.embed.weight if d_model == d_embed else None
            # if dims don't match, use a projection before tying, or skip tying

    def forward(self, x):
        # x: [B, T] token ids, already shifted (input = x[:, :-1], target = x[:, 1:])
        B, T = x.shape
        e = self.embed(x).transpose(1,2)                      # [B, d_embed, T]
        h0 = torch.zeros(B, self.step.candidate.out_features - e.shape[1], T, device=x.device)
        s = torch.cat([e, h0], dim=1)                          # [B, d_model, T]
        for k in range(self.K):
            s = self.step(s, step_idx=k)
        logits = self.readout(s.transpose(1,2))                # [B, T, vocab]
        return logits
```

This preserves **parallel training** — every micro-step is a batched causal conv over the whole sequence at once (like WaveNet/causal-CNN), not a token-by-token loop. This matters: it's the property that made Transformers beat RNNs on training throughput, and NCA-LM keeps it. Say this explicitly in any writeup; it pre-empts the "isn't this just a slow RNN" objection.

### 6.4 Hyperparameter sweep (Level 2 core experiment)

Given TextNCA's finding of a clean optimum around K≈4 with U-shaped degradation beyond it, don't blindly run K up to 32 as originally planned — that's now a **confirmatory** sweep, not exploratory:

| Axis | Values to test | Expected outcome (prior) | What a surprise would mean |
|---|---|---|---|
| K (depth) | 1, 2, 4, 8, 16 | perplexity improves to ~K=4, degrades after | if monotonic improvement continues past 8, this diverges from TextNCA — investigate why (different gating? different task?) |
| radius | 1, 2, 4, 8 | larger radius ≈ trades off against K for same receptive field | if radius and K aren't roughly interchangeable at fixed r·K, the "receptive field" framing is wrong and needs revision |
| gating on/off | GRU-gated vs. plain additive update | gated version needed for K-benefit to appear at all | if plain additive matches gated, you've found a simplification worth reporting |

**Time budget: 1–1.5 weeks**, mostly on WikiText-2, with a confirmation run on WikiText-103 for the single best config only (don't grid-search at the expensive scale).

**Expected headline result: NCA-LM loses on raw perplexity to Transformer and likely to Mamba/RWKV too.** This is the calibrated-expectation outcome per CellARC and TextNCA. Do not spend extra weeks trying to close this gap — that's the KdV/CNN-beats-NCA-by-30x pattern repeating. Move to §7 and Level 3/4 once this is characterized, not once it's "fixed."

---

## 7. The corrected positional-encoding / length-generalization experiment (do this early — it's cheap)

This should run in parallel with Level 1, not after Level 2 — it only needs the Transformer baseline plus NCA-LM once a minimal version exists.

**Protocol:**
1. Train each model at a fixed short length (e.g., 128 tokens).
2. Evaluate perplexity at held-out lengths: 128 (in-distribution), 256, 512, 1024 (extrapolation).
3. Plot perplexity vs. eval length for each configuration:

| Model | Config |
|---|---|
| Transformer | absolute positional encoding |
| Transformer | no positional encoding (causal, "NoPos") |
| NCA-LM | (no PE concept applies — topology is inherent) |
| Mamba/RWKV | (also inherently order-aware via recurrence — include as reference) |

**Why this design, not same-length PE on/off:** causal Transformers already recover implicit absolute position from the causal attention mask alone (Haviv et al., 2022) and are competitive with explicit PE at training length — so an in-distribution "PE vs no-PE" comparison will show little difference and won't test what you think it's testing. The literature's actual finding is that **NoPE Transformers generalize weakly to lengths beyond training** (Kazemnejad et al., 2023: e.g., train ~20 tokens, test ~40, degrades sharply) despite being fine in-distribution. That's the sharp, quantified, already-established failure curve to test NCA-LM against.

**Hypothesis to test:** NCA-LM's fixed local topology is structurally closer to *relative* position encoding (à la ALiBi/RoPE, which are known to generalize better with length than absolute PE or NoPE) rather than "no position information" — because a cell's neighbor relationship (i−1 ↔ i ↔ i+1) is baked into the architecture, not learned. If NCA-LM's degradation curve at 4×–8× training length is flatter than both Transformer variants, that's your sharpest, most citable single result, and it has a mechanistic story (structural relative locality) rather than being an unexplained curiosity.

**Time budget: 3–4 days**, cheap because it reuses Level 1/2 checkpoints and only extends eval, not training.

---

## 8. Level 3 — Hybrid (NCA as adaptor, not replacement)

Directly analogous to AdaNCA (NCA inserted between ViT layers, NeurIPS 2024) but applied to language:

```
Transformer block → NCA-adaptor (few causal micro-steps) → Transformer block → ...
```

- Insert a lightweight NCA step (K=2–4, small radius) between 2–3 selected Transformer layers.
- Target parameter overhead: <5%, matching AdaNCA's claimed efficiency.
- **Evaluate on two axes, not perplexity alone:**
  1. Standard perplexity delta (expect small, possibly negative — that's fine, it's not the point).
  2. **Robustness**: character-level noise injection (typos, swaps, deletions) at test time, measuring perplexity degradation with vs. without the NCA adaptor. This mirrors AdaNCA's adversarial/OOD robustness framing, translated to text corruption, since that's the property their vision result actually demonstrated.

**Time budget: 1–2 weeks.** This is your strongest candidate for a genuinely novel, publishable result, because it's the "augment before replace" move — historically the one that actually worked (Bahdanau attention before Transformers) — and it's testing a hypothesis (text robustness) that AdaNCA's own paper didn't cover.

---

## 9. Level 4 — targeted streaming/persistent-state experiment

Only run this if Level 2 or 3 showed *something* interesting on state persistence (per your KdV memory-swap precedent) — otherwise this is premature.

**Protocol, direct descendant of your KdV memory-swap ablation:**
1. Generate a long document in streaming fashion: feed tokens one at a time, let NCA-LM evolve its persistent cellular state incrementally rather than recomputing over a growing window.
2. Compare against: Transformer with KV-cache (standard incremental decoding), and the Mamba/RWKV baseline (native streaming).
3. Metrics: (a) perplexity as a function of stream length — does NCA-LM's quality degrade, plateau, or stay flat as the "context" grows past what it saw in training; (b) wall-clock latency and memory per new token, across all three architectures.
4. **Corruption/robustness probe**, reusing your KdV technique directly: mid-stream, zero or randomize the persistent hidden state and observe how generation quality recovers or fails, exactly as you did for the memory-swap experiment on KdV. This is a genuine methodological reuse across domains — worth stating explicitly as a validated technique carried over from Phase 1.

**Time budget: 1–2 weeks**, contingent on Level 2/3 results being interesting enough to justify it.

---

## 10. Level 5 — conditional architectural replacement

Explicitly gated: only attempt if Levels 2–4 produced a clear, specific advantage (not just "NCA is interesting"). Given CellARC and TextNCA precedent, budget for this **not happening** in the current project cycle. If it's warranted, scope it as "how much of the Transformer can be replaced while retaining property X" (X = the specific thing that worked, e.g. length generalization or streaming robustness) — not a full architecture race.

---

## 11. Compute budget (rough, single-GPU assumption, ~24GB class card)

| Phase | Wall-clock | Notes |
|---|---|---|
| Infra + Level 0 | 2–3 days | mostly data/tokenizer pipeline |
| Level 1 (3 baselines) | 3–5 days | parallelizable if multi-GPU available |
| §7 PE/length-generalization | 3–4 days | reuses Level 1 checkpoints |
| Level 2 (NCA-LM + sweep) | 1–1.5 weeks | WikiText-2 sweep, single WikiText-103 confirmation run |
| Level 3 (hybrid) | 1–2 weeks | includes robustness eval harness build |
| Level 4 (streaming, conditional) | 1–2 weeks | only if gated-in |
| **Total (through Level 3)** | **~4–5 weeks** | Level 4/5 conditional, add 2–4 more weeks if triggered |

---

## 12. Decision gates (write these in the README, check them honestly)

- **After Level 2 sweep:** if no config beats the K=1 baseline by a meaningful margin, and the K-vs-perplexity curve doesn't even qualitatively match TextNCA's reported shape, stop and diagnose before proceeding — don't grid-search harder, per the KdV lesson about not "optimizing gamma" indefinitely.
- **After §7:** this is your cheapest, highest-signal experiment. If NCA-LM's extrapolation curve is *not* flatter than NoPE-Transformer's, that's a real negative result — report it as such (the structural-locality-as-relative-PE hypothesis was wrong), don't quietly drop the experiment from the writeup.
- **Before Level 4:** require a specific, named signal from Level 2/3 (not vibes) before investing 1–2 weeks in the streaming harness. "It seemed like state mattered" is not a gate-pass; a quantified persistent-state effect (like your KdV memory-swap distortion numbers) is.

---

## 13. Logging / reproducibility

- One run = one config file + one output directory containing: final metrics, param count, git commit hash, and the exact command used.
- Append-only decision log in the README (mirror the style of your KdV summary — the negative results were the most useful part of that log, don't sanitize them out here either).
- Track every "expected per literature X" claim in this plan against your actual result explicitly — a table of predicted vs. observed at the end of each level is cheap to produce and is exactly what makes the final writeup credible.
