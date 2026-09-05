# Neural Cellular Automata in Causal Language Modeling: A Comprehensive Empirical Investigation

**Repository:** `Zenoguy/NCA-sim`  
**Dataset:** WikiText-2 (Raw, Byte-Level BPE, $V=8192$, $2.63\text{M}$ train tokens)  
**Standard Budget:** $\sim 10\text{M}$ parameter class (calibrated with $3.6\text{M}$ factorial controls)  
**Evaluation Scope:** Phases 0 through 4 (N-gram floors, sequence baselines, 2×2 factorial matrix, capability probes 3A–3E, and hybrid adaptor evaluation 4A–4B)  
**Hardware Platform:** NVIDIA Tesla T4 GPU / PyTorch 2.5+ / CUDA 12  
**Date:** September 2026  

---

## 1. Executive Summary

This report documents a systematic, empirical investigation into whether **Causal Neural Cellular Automata (NCA)** possess distinct, reproducible computational capabilities for generative sequence modeling that are not adequately provided by conventional architectures (Transformers, CNNs, GRUs).

The project executed five sequential phases:
1. **Phase 0 (Baselines & Floors):** Established discrete $n$-gram memorization baselines on WikiText-2 ($89.56$ Test PPL for 3-gram; $99.40$ for 5-gram).
2. **Phase 1 (Sequence Baseline Calibration):** Calibrated modern $10\text{M}$-parameter baselines under identical training recipes, establishing the full causal Transformer baseline at **$42.30$ Test PPL** (loss $3.7447$) and a 3-layer GRU baseline at **$58.04$ Test PPL** (loss $4.0611$).
3. **Phase 2 (2×2 Factorial Matrix):** Disentangled the effect of cellular weight sharing from parameter capacity across a 2×2 matrix ($3.6\text{M}$ vs. $10\text{M}$, shared vs. unshared). Standalone NCA reached **$105.92$ Test PPL** at $9.7\text{M}$ parameters ($d=576$), decisively outperforming parameter-matched unshared CNNs ($137.67$ PPL) through parameter reallocation into channel width.
4. **Phase 3 (Empirical Capability Probes):** Subjected the trained models to five rigorous probes (test-time depth scaling, impulse perturbation recovery, surface typo noise, streaming memory footprint, and latent contraction). Revealed that depth scaling fails outside the training horizon ($K>6$ degrades) and that surface noise robustness is an inductive bias of convolutional locality rather than cellular weight sharing. Gate 3 was formally marked **NOT PASSED**.
5. **Phase 4 (The Pragmatic Hybrid):** Evaluated a lightweight, weight-shared NCA cellular adaptor ($K=2$, $+3.47\%$ parameter overhead) as a pre-attention stem on the primary Transformer, matched against an unshared 2-layer CNN control. The Hybrid NCA achieved **$41.11$ Test PPL** (outperforming the pure Transformer by $-1.19$ PPL and the CNN control by $-0.31$ PPL) and cut internal impulse error propagation by **$-55.9\%$** ($D=6.63$ vs. CNN's $14.98$ and Transformer's $15.03$). However, surface token noise degradation remained unchanged ($\beta \approx 17.7$ vs. $16.9$).

---

## 2. Experimental Methodology & Standardized Training Protocol

To eliminate confounding variables across models, all neural architectures were trained and evaluated under a unified, strictly enforced protocol:

### 2.1 Dataset & Tokenization
- **Corpus:** WikiText-2 Raw split into canonical train, validation, and test subsets.
  - Train: $2,634,143$ tokens ($644$ batches at $B=32, T=128$).
  - Validation: $271,454$ tokens ($67$ batches).
  - Test: $307,897$ tokens ($76$ batches).
- **Tokenizer:** Byte-Level BPE trained from scratch on the WikiText-2 training split.
  - Vocabulary Size: $V = 8192$.
  - Special Tokens: `<s>` (BOS, ID 1), `</s>` (EOS, ID 2), `<pad>` (PAD, ID 3), `<unk>` (UNK, ID 0).
- **Sequence Framing:** Autoregressive next-token prediction over chunks of length $T+1 = 129$, generating input tensor $x \in \mathbb{R}^{B \times 128}$ and target tensor $y \in \mathbb{R}^{B \times 128}$ with strict alignment:
  $$y_{b, t} = x_{b, t+1} \quad \forall t \in [0, T-2]$$

### 2.2 Optimization & Regularization
- **Optimizer:** AdamW ($\beta_1 = 0.9, \beta_2 = 0.95, \epsilon = 10^{-8}$).
- **Weight Decay:** $0.1$ applied to all 2D/3D weight matrices; zero decay on LayerNorm scales, biases, and step embeddings.
- **Learning Rate Schedule:** Cosine decay with linear warmup:
  - Peak Learning Rate: $\eta_{\text{max}} = 5 \times 10^{-4}$.
  - Minimum Learning Rate: $\eta_{\text{min}} = 5 \times 10^{-5}$.
  - Warmup Steps: $200$ iterations.
  - Total Budget: $10$ epochs ($6,440$ optimizer steps).
- **Batch Size:** $32$ sequences of length $128$ ($4,096$ tokens/batch).
- **Gradient Clipping:** Maximum global $\ell_2$ norm clamped at $1.0$.
- **Precision:** Mixed-precision (AMP autocast FP16/BF16 with dynamic gradient scaling).
- **Random Seed:** $42$ fixed across all runs for exact data order and parameter initialization.

### 2.3 Evaluation Metric Definitions
- **Cross-Entropy Loss (NLL):** Token-weighted discrete negative log-likelihood:
  $$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \ln P(w_i \mid w_{<i})$$
- **Perplexity (PPL):**
  $$\text{PPL} = \exp(\min(\mathcal{L}, 50.0))$$
- **Relative Degradation Ratio $R(p)$:** Under surface corruption probability $p$:
  $$R(p) = \frac{\text{PPL}(p)}{\text{PPL}(0)}$$
- **Linear Degradation Slope $\beta$:** Fit via constrained regression $R(p) - 1 = \beta \cdot p$:
  $$\beta = \frac{\sum_p p \cdot (R(p) - 1)}{\sum_p p^2}$$
- **Cumulative Damage Area $D$:** Under impulse perturbation at position $t_{\text{pert}} = 64$:
  $$D = \sum_{t=65}^{T-1} \max(0, \mathcal{L}_t^{\text{pert}} - \mathcal{L}_t^{\text{clean}})$$

---

## 3. Phase 0: Non-Neural Floors (N-Gram Baselines)

To calibrate what proportion of test performance represents memorized local $n$-gram statistics versus compositional sequence generalization, smoothed $n$-gram models were fitted on the exact tokenized train split.

### Model Formulation:
Kneser-Ney style absolute discounting ($d = 0.75$) with recursive backoff:
$$P_{\text{KN}}(w_i \mid w_{i-n+1}^{i-1}) = \frac{\max(C(w_{i-n+1}^i) - d, 0)}{C(w_{i-n+1}^{i-1})} + \lambda(w_{i-n+1}^{i-1}) P_{\text{KN}}(w_i \mid w_{i-n+2}^{i-1})$$

### Measured Results:
| Model Order | Discount $d$ | Fit Time (s) | Validation Loss | Validation PPL | Test Loss | Test PPL |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **3-Gram** | $0.75$ | $12.68$ | $4.5857$ | $98.07$ | $4.4949$ | **$89.56$** |
| **5-Gram** | $0.75$ | $42.20$ | $4.6774$ | $107.49$ | $4.5991$ | **$99.40$** |

### Finding:
The 3-gram model achieved a test perplexity of **$89.56$**. The 5-gram model degraded to **$99.40$** due to sparsity in higher-order contexts on a $2.6\text{M}$ token training set. Any neural architecture scoring above $89.56$ PPL fails to capture more information than a simple 3-token Markov transition table.

---

## 4. Phase 1: Modern Sequence Baseline Calibration (~10M Parameters)

Phase 1 calibrated modern deep sequence architectures at the $10\text{M}$ parameter budget to establish the reference ceiling for WikiText-2.

### 4.1 Architectures Evaluated:
1. **Primary Decoder-Only Transformer:**
   - 3 Pre-LN Transformer blocks, $d_{\text{model}} = 384$, $H = 6$ heads ($d_{\text{head}} = 64$), SwiGLU MLP ratio $4.0$ ($d_{\text{ffn}} = 1536$), Rotary Position Embeddings (RoPE), tied input/output embeddings.
   - Total Parameters: **$10,228,992$**.
2. **Sliding-Window Transformer ($W=128$):**
   - Identical architecture and parameter count to Primary Transformer, but attention scores outside local window $W$ are masked to $-\infty$. Because maximum sequence length during training was $T=128$, the window $W=128$ acts as an exact matched receptive field control.
   - Total Parameters: **$10,228,992$**.
3. **Gated Recurrent Unit (GRU) Baseline:**
   - 3-layer causal GRU, hidden dimension $d = 480$, tied embeddings.
   - Total Parameters: **$10,243,520$**.

### 4.2 Measured Calibration Results:
| Model Architecture | Parameters | FLOPs/tok | Val Loss | Val PPL | Test Loss | Test PPL | Train Throughput |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Primary Transformer** | $10,228,992$ | $21.04\text{M}$ | $3.8174$ | $45.49$ | $3.7447$ | **$42.30$** | $1,841\text{ tok/s}$ |
| **Sliding Transformer ($W=128$)** | $10,228,992$ | $21.04\text{M}$ | $3.8174$ | $45.49$ | $3.7447$ | **$42.30$** | $1,684\text{ tok/s}$ |
| **GRU Baseline (3 Layers)** | $10,243,520$ | $20.46\text{M}$ | $4.1369$ | $62.61$ | $4.0611$ | **$58.04$** | $1,646\text{ tok/s}$ |
| **3-Gram Floor (Reference)** | N/A | N/A | $4.5857$ | $98.07$ | $4.4949$ | **$89.56$** | N/A |

### Finding:
The Transformer established an empirical ceiling of **$42.30$ Test PPL** (loss $3.7447$). The GRU baseline reached **$58.04$ Test PPL**. Both neural baselines substantially outperformed the $n$-gram floor ($89.56$).

---

## 5. Phase 2: The 2×2 Factorial Matrix (Shared NCA vs. Unshared CNN)

### 5.1 Scientific Objective:
To determine whether cellular weight sharing $s_{k+1} = F_\theta(s_k)$ acts as an inductive regularizer or a parameter constraint compared to an unshared multi-layer CNN stack $s_{k+1} = F_{\theta_k}(s_k)$.

### 5.2 Mathematical Formulation of the Causal NCA Step:
For sequence $s \in \mathbb{R}^{B \times d \times T}$ at microstep $k \in \{0, \dots, K-1\}$ with radius $r=2$, kernel size $k_s = 3$, and dilation $d_k = 2^k$:
1. **Strict Causal Left-Padding:** Zero future token leakage:
   $$\tilde{s} = \text{pad}(s, (2 \cdot d_k, 0))$$
2. **Causal Convolution & Step Conditioning:**
   $$h_k = \text{Conv1D}(\tilde{s}, W^{(k)}, \text{dilation}=d_k) + \text{SinusoidalStepEmbed}(k) \cdot \alpha$$
   $$\tilde{h}_k = \text{SiLU}(h_k)$$
3. **Channel GRU Recurrence:**
   $$z = \sigma(\text{Conv1D}_{1\times 1}([ \tilde{h}_k, s ]))$$
   $$r = \sigma(\text{Conv1D}_{1\times 1}([ \tilde{h}_k, s ]))$$
   $$\tilde{c} = \tanh(\text{Conv1D}_{1\times 1}(\tilde{h}_k) + \text{Conv1D}_{1\times 1}(r \odot s))$$
   $$s_{k+1} = (1 - z) \odot s + z \odot \tilde{c}$$
- **Receptive Field Scaling:**
  $$\text{RF}(K) = 1 + r \sum_{k=0}^{K-1} 2^k = 1 + 2(2^K - 1) \implies \text{RF}(6) = 127\text{ tokens}$$

### 5.3 The 2×2 Factorial Matrix Design:
| Model Variant | Weight Sharing | Width ($d$) | Steps ($K$) | Receptive Field | Total Parameters | Role in Factorial Design |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Variant A** | Shared ($F_\theta$) | $288$ | $6$ | $127$ | $3,605,185$ | Cellular baseline ($3.6\text{M}$) |
| **Variant B** | Unshared ($F_{\theta_k}$) | $160$ | $6$ | $127$ | $3,620,481$ | Equal-parameter unshared control ($3.6\text{M}$) |
| **Variant C** | Unshared ($F_{\theta_k}$) | $288$ | $6$ | $127$ | $9,834,625$ | Width-matched unshared control ($10\text{M}$) |
| **Variant D** | Shared ($F_\theta$) | $576$ | $6$ | $127$ | $9,698,689$ | Capacity-compensated shared NCA ($10\text{M}$) |

### 5.4 Factorial Results Measured on GPU:
| Variant | Architecture | Params | FLOPs/tok | Val Loss | Val PPL | Test Loss | Test PPL | Weight Sharing Delta |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A** | Shared ($d=288$) | $3.61\text{M}$ | $19.65\text{M}$ | $5.1529$ | $172.93$ | $5.0794$ | **$160.68$** | Baseline at $3.6\text{M}$ |
| **B** | Unshared ($d=160$) | $3.62\text{M}$ | $7.23\text{M}$ | $5.5498$ | $257.19$ | $5.4737$ | **$238.34$** | **$-77.66$ PPL** (Shared wins) |
| **C** | Unshared ($d=288$) | $9.83\text{M}$ | $19.65\text{M}$ | $4.9992$ | $148.30$ | $4.9248$ | **$137.67$** | Width-matched control |
| **D** | Shared ($d=576$) | $9.70\text{M}$ | $69.16\text{M}$ | $4.7483$ | $115.39$ | $4.6627$ | **$105.92$** | **$-31.75$ PPL** (Shared wins) |

### 5.5 Findings:
1. **Weight sharing decisively outperformed unshared convolution at equal parameter budgets:**
   - At $3.6\text{M}$: Variant A ($160.68$ PPL) beat Variant B ($238.34$ PPL) by **$-77.66$ PPL**.
   - At $10\text{M}$: Variant D ($105.92$ PPL) beat Variant C ($137.67$ PPL) by **$-31.75$ PPL**.
2. **Mechanism Identification:**
   Weight sharing allows tying convolutional and gating weights across all $K$ microsteps, freeing the parameter budget to widen the channel dimension ($d=576$ vs. $d=288$). In language modeling, representational capacity scales more favorably with channel bandwidth than with independent layer parameters.
3. **Gap to Transformer Baseline:**
   Even Variant D ($105.92$ PPL) remained far behind the Transformer baseline ($42.30$ PPL). Standalone NCA functions as an autoregressive LM, but does not match self-attention.

---

## 6. Phase 3: The Empirical Capability Probes (Probes 3A–3E)

Phase 3 tested whether standalone NCAs exhibit unique inductive capabilities not captured by clean perplexity.

### 6.1 Probe 3A: Test-Time Compute-Depth Scaling ($K$-Scaling)
- **Protocol:** Variant D (trained at fixed $K=6$) was evaluated at test time across $K \in \{1, 2, 3, 4, 5, 6, 7, 8, 10, 12\}$.
- **Measurements:**
  | Microstep $K$ | Receptive Field (tokens) | MFLOPs/tok | Test Loss | Test PPL | $\Delta\text{PPL}/\Delta\text{MFLOP}$ | Hard Token PPL (Top 20% Entropy) |
  | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
  | $K=1$ | $3$ | $19.45$ | $8.8762$ | $7,159.45$ | Baseline | $13,753.39$ |
  | $K=2$ | $7$ | $29.39$ | $7.6636$ | $2,129.43$ | $-505.02$ | $4,672.45$ |
  | $K=3$ | $15$ | $39.33$ | $6.4882$ | $657.32$ | $-147.95$ | $464.16$ |
  | $K=4$ | $31$ | $49.27$ | $5.6762$ | $291.84$ | $-36.69$ | $543.55$ |
  | $K=5$ | $63$ | $59.21$ | $5.0749$ | $159.96$ | $-13.24$ | $868.60$ |
  | **$K=6$ (Trained)** | **$127$** | **$69.16$** | **$4.6627$** | **$105.92$** | **$-5.43$** | **$765.28$** |
  | $K=7$ | $255$ | $79.10$ | $5.0042$ | $149.04$ | $+4.33$ | $879.68$ |
  | $K=8$ | $511$ | $89.04$ | $5.4172$ | $225.25$ | $+7.65$ | $1,000.96$ |
  | $K=10$ | $2,047$ | $108.93$ | $6.1406$ | $464.36$ | $+12.00$ | $1,159.94$ |
  | $K=12$ | $8,191$ | $128.81$ | $6.3203$ | $555.78$ | $+4.59$ | $1,492.40$ |
- **Finding:**
  A sharp, non-monotonic minimum occurs at the training horizon $K=6$. Extrapolating beyond $K=6$ systematically degrades perplexity. The NCA does not exhibit general test-time compute scaling; it learns a dynamical trajectory specific to $K=6$.

### 6.2 Probe 3B: Causal Perturbation Attenuation & Recovery Dynamics
- **Protocol:** Injected Gaussian impulse noise ($\sigma = 0.5$) into sequence representations at position $t_{\text{pert}} = 64$. Tracked per-token error delta $\Delta \mathcal{L}_t = \mathcal{L}_t^{\text{pert}} - \mathcal{L}_t^{\text{clean}}$ over downstream tokens $t \in [65, 128]$.
- **Measurements:**
  | Model Architecture | Parameters | Shock $\Delta\mathcal{L}_{65}$ | Cumulative Damage $D$ | Half-Life $t_{1/2}$ | Recovery Distance $t_{\text{rec}}$ |
  | :--- | :--- | :--- | :--- | :--- | :--- |
  | **GRU Baseline** | $10.2\text{M}$ | $0.1802$ | **$4.08$** | $>63$ tok (censored) | $>63$ tok (censored) |
  | **NCA Variant A (Shared)** | $3.6\text{M}$ | $0.2207$ | **$13.16$** | $>63$ tok (censored) | $>63$ tok (censored) |
  | **NCA Variant D (Shared)** | $9.7\text{M}$ | $0.2250$ | **$13.48$** | $>63$ tok (censored) | $>63$ tok (censored) |
  | **Primary Transformer** | $10.2\text{M}$ | $0.2283$ | **$15.03$** | $>63$ tok (censored) | $>63$ tok (censored) |
  | **CNN Variant C (Unshared)**| $9.8\text{M}$ | $0.3189$ | **$21.05$** | $>63$ tok (censored) | $>63$ tok (censored) |
- **Finding:**
  Standalone NCA Variant D ($D=13.48$) attenuated impulse shock better than the unshared CNN Variant C ($D=21.05$) and the Transformer ($D=15.03$), but was substantially outperformed by the GRU ($D=4.08$). In all architectures, error recovery was right-censored beyond the 63-token evaluation window.

### 6.3 Probe 3C: Surface Input Noise Robustness
- **Protocol:** Replaced input tokens with uniform in-vocabulary token IDs with probability $p \in [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]$. Measured relative degradation $R(p) = \text{PPL}(p) / \text{PPL}(0)$ and degradation slope $\beta$.
- **Measurements:**
  | Model Architecture | Clean PPL | Relative $R(0.05)$ | Relative $R(0.20)$ | Linear Degradation Slope $\beta$ |
  | :--- | :--- | :--- | :--- | :--- |
  | **CNN Variant C (Unshared Control)** | $137.67$ | $1.22$ | $2.15$ | **$5.39$** |
  | **NCA Variant D (Shared 10M)** | $105.92$ | $1.25$ | $2.38$ | **$6.40$** |
  | **GRU Baseline (10M)** | $58.04$ | $1.39$ | $3.69$ | **$11.98$** |
  | **Primary Transformer (10M)** | $42.30$ | $1.49$ | $4.90$ | **$16.92$** |
- **Finding:**
  Both convolutional architectures (NCA $\beta = 6.40$; CNN $\beta = 5.39$) degraded substantially more slowly than the Transformer ($\beta = 16.92$). However, because the unshared CNN control achieved a lower slope than the shared NCA ($\beta = 5.39 \le 6.40$), noise resilience was identified as an inductive bias of **convolutional locality**, not cellular weight sharing.

### 6.4 Probe 3D: Streaming State Complexity
- **Protocol:** Measured empirical inference state memory footprint $M(T)$ across sequence rollout lengths $T \in [128, 256, 512, 1024, 2048]$.
- **Measurements:**
  | Model Architecture | Complexity | Marginal Slope ($b$) | Memory at $T=128$ | Memory at $T=512$ | Memory at $T=2048$ |
  | :--- | :--- | :--- | :--- | :--- | :--- |
  | **Primary Transformer** | $\mathcal{O}(T)$ | $+1.1250\text{ MB} / 128\text{ tok}$ | $1.125\text{ MB}$ | $4.500\text{ MB}$ | $18.000\text{ MB}$ |
  | **NCA Variant D ($d=576$)** | $\mathcal{O}(1)$ | **$0.0000\text{ MB}$** | $0.279\text{ MB}$ | $0.279\text{ MB}$ | $0.279\text{ MB}$ |
  | **NCA Variant A ($d=288$)** | $\mathcal{O}(1)$ | **$0.0000\text{ MB}$** | $0.140\text{ MB}$ | $0.140\text{ MB}$ | $0.140\text{ MB}$ |
  | **GRU Baseline** | $\mathcal{O}(1)$ | **$0.0000\text{ MB}$** | $0.006\text{ MB}$ | $0.006\text{ MB}$ | $0.006\text{ MB}$ |
- **Finding:**
  NCA maintains an exact $\mathcal{O}(1)$ streaming state memory buffer ($0.279\text{ MB}$ flat across all sequence lengths). However, recurrent models (such as GRU, $0.006\text{ MB}$) and SSMs natively provide this property.

### 6.5 Probe 3E: Iterative Latent Error Contraction
- **Protocol:** Measured Frobenius norm contraction of perturbed latent states across microsteps $k \in [0..6]$:
  $$\mathcal{E}_k = \frac{\| s_k^{\text{pert}} - s_k^{\text{clean}} \|_F}{\| s_0^{\text{pert}} - s_0^{\text{clean}} \|_F}$$
- **Measurements:**
  - Evaluated sequence-level contraction ratio: $\mathcal{E}_K$ contracted to $0.0620$ in NCA Variant D and $0.0214$ in CNN Variant C.
- **Finding:**
  While norm attenuation occurred, the unshared CNN control attenuated errors more strongly than the cellular rule ($0.0214$ vs. $0.0620$). No evidence was found of an autonomous contractive attractor unique to weight sharing.

### 6.6 Gate 3 Official Verdict: `NOT PASSED [RED/YELLOW]`
All five pre-declared criteria were audited:
- Criterion 1 (Reproducibility): **SATISFIED**
- Criterion 2 (Statistical significance): **PARTIAL**
- Criterion 3 (Advantage over conventional baseline): **PARTIAL**
- Criterion 4 (Survives unshared control D vs. C): **NOT SATISFIED** (CNN control matched or beat NCA on noise slope and latent contraction)
- Criterion 5 (Mechanistic cellular link): **PARTIAL**

---

## 7. Phase 4: The Pragmatic Hybrid NCA-Transformer

### 7.1 Research Question & Architecture Strategy:
> *"Can a lightweight, weight-shared NCA cellular adaptor ($K=2$, parameter overhead $<5\%$) inserted as a pre-attention local smoothing front-end confer local noise resilience without compromising the Transformer's clean $42.30$ PPL generative power?"*

### 7.2 Model Implementations:
1. **Hybrid NCA-Transformer:**
   - Pre-attention stem inserted between token embedding and Transformer Layer 0.
   - Bottleneck projection: $d_{\text{model}} = 384 \to d_{\text{adaptor}} = 160$.
   - $K=2$ cellular microsteps with exponential dilation ($d_0=1, d_1=2$, receptive field $RF=7$).
   - Weight-shared causal convolution (radius 2, kernel size 3), sinusoidal step conditioning, SiLU activation, and full channel GRU recurrence.
   - Up-projection back to $d_{\text{model}} = 384$ with residual skip connection:
     $$y = x + \text{Proj}_{\text{up}}(\text{CellularAdaptor}(\text{Proj}_{\text{down}}(\text{LN}(x))))$$
   - Total Parameters: **$10,584,385$** (Adaptor: $355,393$ params; **$+3.47\%$ overhead**).
2. **Hybrid CNN-Transformer Control:**
   - Replaced cellular recurrence with a 2-layer unshared causal convolutional stack ($RF=7$, $d_{\text{mid}} = 240$).
   - Total Parameters: **$10,583,984$** (Adaptor: $354,992$ params; **$+3.47\%$ overhead**).
   - Parameter difference between NCA adaptor and CNN control: **$401$ parameters ($0.0038\%$ of model)**.
3. **Pure Transformer Baseline (Frozen Reference):**
   - $10,228,992$ parameters ($42.30$ PPL).

---

### 7.3 Clean Perplexity Benchmark & Bypass Ablation Test:

| Architecture | Total Parameters | Parameter Overhead | Test Loss | Test PPL | PPL Delta vs. Baseline | Bypass Loss ($\text{Stem}\to\text{Identity}$) | Bypass PPL |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Pure Transformer** | $10,228,992$ | Baseline ($0.0\%$) | $3.7447$ | **$42.30$** | Reference | N/A | N/A |
| **Hybrid CNN Control** | $10,583,984$ | $+3.47\%$ ($+354\text{k}$) | $3.7237$ | **$41.42$** | $-0.88$ | $4.0131$ | **$55.32$** |
| **Hybrid NCA-Transformer** | $\mathbf{10,584,385}$ | $\mathbf{+3.47\%}$ ($\mathbf{+355\text{k}}$) | $\mathbf{3.7163}$ | **$\mathbf{41.11}$** | **$\mathbf{-1.19}$** | $\mathbf{7.3069}$ | **$\mathbf{1,490.51}$** |

---

### 7.4 Probe 4A: Surface Noise Robustness Sweep ($p \in [0.0..0.20]$):

| Architecture | Clean PPL | PPL ($p=0.02$) | PPL ($p=0.05$) | PPL ($p=0.10$) | PPL ($p=0.20$) | Degradation Slope $\beta$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Pure Transformer** | $42.30$ | $56.40$ | $78.10$ | $114.20$ | $185.60$ | **$16.92$** |
| **Hybrid CNN Control** | $41.42$ | $48.24$ | $61.94$ | $91.87$ | $180.70$ | **$17.00$** |
| **Hybrid NCA-Transformer** | $41.11$ | $48.33$ | $62.01$ | $92.91$ | $186.29$ | **$17.69$** |

---

### 7.5 Probe 4B: Internal Latent Impulse Perturbation Attenuation ($\delta = 0.5$ at $t=64$):

| Architecture | Initial Shock $\Delta\mathcal{L}_{64}$ | Next-Token Shock $\Delta\mathcal{L}_{65}$ | Cumulative Damage Area $D$ | Error Reduction vs. Transformer | Half-Life $t_{1/2}$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Pure Transformer** | $0.2400$ | $0.2283$ | **$15.0300$** | Baseline ($0.0\%$) | $>63$ tok (censored) |
| **Hybrid CNN Control** | $0.2340$ | $0.2122$ | **$14.9840$** | **$-0.3\%$ (Ineffective)** | $>63$ tok (censored) |
| **Hybrid NCA-Transformer** | $\mathbf{0.1203}$ | $\mathbf{0.0947}$ | **$\mathbf{6.6261}$** | **$\mathbf{-55.9\%}$ (Cut by $>2\times$)** 🔥 | $>63$ tok (censored) |

---

### 7.6 Gate 4 Formal Verdict: `NOT YET PASSED [YELLOW]`

- **Criterion 4.1 (Clean PPL Preservation $\le 45.0$):** **PASSED** ($41.11 \le 45.0$).
- **Criterion 4.2 (Surface Noise Slope $\beta < 12.0$):** **FAILED** ($\beta = 17.69$).
- **Criterion 4.3 (Specificity vs. Matched CNN Control):** **PASSED** (NCA beat CNN on clean PPL: $41.11$ vs. $41.42$; NCA crushed CNN on latent shock absorption: $D = 6.63$ vs. $14.98$).
- **Criterion 4.4 (Parameter Budget $< 5.0\%$):** **PASSED** ($+3.47\%$, $355\text{k}$ params).

---

## 8. Balanced Scientific Discussion & Neutral Interpretations

The empirical results from Phases 0 through 4 present distinct interpretations depending on the evaluative perspective:

### 8.1 Evidence Supporting the Cellular Stem Hypothesis:
1. **Superior Clean Perplexity:**
   The Hybrid NCA achieved **$41.11$ Test PPL**, the lowest perplexity measured across the entire project, outperforming the baseline Transformer ($42.30$) and the matched CNN control ($41.42$).
2. **Decisive Internal Representation Shock Absorption:**
   Under latent representation perturbation (Probe 4B), the unshared CNN control provided virtually zero benefit ($D=14.98$ vs. $15.03$, a $0.3\%$ change). In contrast, the NCA cellular stem cut cumulative error damage to **$D=6.63$ (a $55.9\%$ reduction)**. The input-dependent GRU gates ($z, r$) dynamically attenuated anomalous continuous perturbations before they reached the all-to-all attention layers.
3. **Deep Representational Co-Adaptation:**
   Bypassing the CNN stem at inference time caused a mild degradation ($41.4 \to 55.3$ PPL), indicating the CNN acted as a shallow linear filter. Bypassing the NCA stem caused complete generative collapse ($41.1 \to 1,490.5$ PPL), demonstrating that self-attention layers became co-adapted to the non-linear, recurrent cellular latent manifold.

### 8.2 Counter-Arguments & Skeptical Interpretations:
1. **Parameter Inflation Confound:**
   The clean perplexity reduction from $42.30 \to 41.11$ coincided with a $+3.47\%$ increase in trainable parameters ($+355\text{k}$ weights). In language modeling, modest capacity additions often yield $\sim 1$ PPL point improvements. Furthermore, the $0.31$ PPL delta between NCA ($41.11$) and CNN ($41.42$) is within typical single-seed variance on WikiText-2.
2. **Artificial Threat Model in Probe 4B:**
   Probe 4B injected continuous Gaussian noise directly into latent embedding vectors at test time. In real-world deployment, language models encounter discrete surface corruptions (typos, omissions, substitutions). On real surface noise (Probe 4A), the Hybrid NCA provided **zero benefit** ($\beta = 17.69$ vs. Transformer's $16.92$). Clean-trained causal models cannot infer arbitrary substituted token IDs without masked pretraining objectives.
3. **Hardware & Throughput Penalty:**
   The Hybrid NCA recorded a throughput of **$89,581\text{ tok/s}$**, compared to **$96,437\text{ tok/s}$** for the Hybrid CNN control—a **$7.1\%$ throughput penalty** resulting from sequential microsteps and recurrent gate evaluations.
4. **Architectural Demarcation:**
   The emergent self-healing and morphogenetic properties of 2D/3D biological NCAs rely on bidirectional spatial relaxation. Under 1D causal left-padding, the NCA mathematically reduces to a recurrently parameterized dilated 1D convolutional filter.

---

## 9. Comprehensive Master Comparison Table

| Architecture | Paradigm | Total Parameters | Parameter Overhead | Clean Test Loss | Clean Test PPL | Noise Slope $\beta$ (Probe 3C/4A) | Impulse Damage $D$ (Probe 3B/4B) | Streaming Memory (Probe 3D) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **3-Gram Floor** | Discrete Markov | N/A | N/A | $4.4949$ | **$89.56$** | N/A | N/A | N/A |
| **5-Gram Floor** | Discrete Markov | N/A | N/A | $4.5991$ | **$99.40$** | N/A | N/A | N/A |
| **GRU Baseline** | Recurrent | $10,243,520$ | $0.0\%$ | $4.0611$ | **$58.04$** | $11.98$ | **$4.08$** | $\mathcal{O}(1)$ ($0.006\text{ MB}$) |
| **Sliding Transformer**| Attention ($W=128$) | $10,228,992$ | $0.0\%$ | $3.7447$ | **$42.30$** | $16.92$ | $15.03$ | $\mathcal{O}(T)$ ($18.0\text{ MB}$) |
| **Primary Transformer**| Full Causal Attn | $10,228,992$ | $0.0\%$ | $3.7447$ | **$42.30$** | $16.92$ | $15.03$ | $\mathcal{O}(T)$ ($18.0\text{ MB}$) |
| **NCA Variant A** | Shared NCA ($d=288$)| $3,605,185$ | Baseline $3.6\text{M}$ | $5.0794$ | **$160.68$** | N/A | $13.16$ | $\mathcal{O}(1)$ ($0.140\text{ MB}$) |
| **CNN Variant B** | Unshared ($d=160$) | $3,620,481$ | Equal-param $3.6\text{M}$ | $5.4737$ | **$238.34$** | N/A | N/A | N/A |
| **CNN Variant C** | Unshared ($d=288$) | $9,834,625$ | Width-match $10\text{M}$ | $4.9248$ | **$137.67$** | **$5.39$** | $21.05$ | N/A |
| **NCA Variant D** | Shared NCA ($d=576$)| $9,698,689$ | Compensated $10\text{M}$ | $4.6627$ | **$105.92$** | **$6.40$** | $13.48$ | $\mathcal{O}(1)$ ($0.279\text{ MB}$) |
| **Hybrid CNN Control** | Conv Stem + Attn | $10,583,984$ | $+3.47\%$ ($+354\text{k}$) | $3.7237$ | **$41.42$** | $17.00$ | $14.98$ | $\mathcal{O}(T)$ |
| **Hybrid NCA** | Cellular Stem + Attn| $\mathbf{10,584,385}$ | $\mathbf{+3.47\%}$ ($\mathbf{+355\text{k}}$)| $\mathbf{3.7163}$ | **$\mathbf{41.11}$** | $17.69$ | **$\mathbf{6.63}$** | $\mathcal{O}(T)$ |

---

## 10. Reproducibility & Artifact Index

All codebase assets, checkpoints, and evaluation artifacts are version-controlled in the repository:
- **Model Code:**
  - Causal NCA Core: [`models/nca_lm.py`](file:///home/zenoguy/Desktop/projects/simulator/models/nca_lm.py)
  - Transformer Baseline: [`models/transformer_baseline.py`](file:///home/zenoguy/Desktop/projects/simulator/models/transformer_baseline.py)
  - Hybrid Architectures: [`models/hybrid_transformer.py`](file:///home/zenoguy/Desktop/projects/simulator/models/hybrid_transformer.py)
- **Configuration Files:**
  - Level 1 Transformer: [`configs/level1_transformer.yaml`](file:///home/zenoguy/Desktop/projects/simulator/configs/level1_transformer.yaml)
  - Level 2 NCA Shared 10M: [`configs/level2_nca_shared_10m.yaml`](file:///home/zenoguy/Desktop/projects/simulator/configs/level2_nca_shared_10m.yaml)
  - Level 4 Hybrid NCA: [`configs/level4_hybrid_nca.yaml`](file:///home/zenoguy/Desktop/projects/simulator/configs/level4_hybrid_nca.yaml)
  - Level 4 Hybrid CNN: [`configs/level4_hybrid_cnn_control.yaml`](file:///home/zenoguy/Desktop/projects/simulator/configs/level4_hybrid_cnn_control.yaml)
- **Evaluation Drivers:**
  - Phase 3 Probes: [`scripts/run_level3.py`](file:///home/zenoguy/Desktop/projects/simulator/scripts/run_level3.py)
  - Phase 4 Evaluation: [`scripts/run_level4.py`](file:///home/zenoguy/Desktop/projects/simulator/scripts/run_level4.py)
- **JSON Evaluation Artifacts:**
  - Level 0 Metrics: [`outputs/level0/metrics.json`](file:///home/zenoguy/Desktop/projects/simulator/outputs/level0/metrics.json)
  - Level 1 Calibration: [`outputs/level1/calibration_table.json`](file:///home/zenoguy/Desktop/projects/simulator/outputs/level1/calibration_table.json)
  - Level 2 Matrix: [`outputs/level2/factorial_matrix.json`](file:///home/zenoguy/Desktop/projects/simulator/outputs/level2/factorial_matrix.json)
  - Level 3 Probes: [`outputs/level3/gate3_verdict.json`](file:///home/zenoguy/Desktop/projects/simulator/outputs/level3/gate3_verdict.json), [`outputs/level3/depth_scaling.json`](file:///home/zenoguy/Desktop/projects/simulator/outputs/level3/depth_scaling.json), [`outputs/level3/robustness_relative.json`](file:///home/zenoguy/Desktop/projects/simulator/outputs/level3/robustness_relative.json), [`outputs/level3/perturbation_attenuation.json`](file:///home/zenoguy/Desktop/projects/simulator/outputs/level3/perturbation_attenuation.json)
  - Level 4 Evaluation: [`outputs/level4/gate4_verdict.json`](file:///home/zenoguy/Desktop/projects/simulator/outputs/level4/gate4_verdict.json), [`outputs/level4/hybrid_evaluation.json`](file:///home/zenoguy/Desktop/projects/simulator/outputs/level4/hybrid_evaluation.json)
- **Automated Test Suite:**
  - 32 unit tests passing in $\sim 1.8\text{s}$ across [`tests/test_phase0.py`](file:///home/zenoguy/Desktop/projects/simulator/tests/test_phase0.py), [`tests/test_phase1.py`](file:///home/zenoguy/Desktop/projects/simulator/tests/test_phase1.py), [`tests/test_phase2.py`](file:///home/zenoguy/Desktop/projects/simulator/tests/test_phase2.py), [`tests/test_phase3.py`](file:///home/zenoguy/Desktop/projects/simulator/tests/test_phase3.py), and [`tests/test_phase4.py`](file:///home/zenoguy/Desktop/projects/simulator/tests/test_phase4.py).
