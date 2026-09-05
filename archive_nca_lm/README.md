# Archive: Causal Neural Cellular Automata Language Model (NCA-LM)

> **Status:** Completed & Archived (Phases 0 through 4)  
> **Primary Report:** [`NCA_LM_Comprehensive_Research_Report.md`](file:///home/zenoguy/Desktop/projects/simulator/archive_nca_lm/NCA_LM_Comprehensive_Research_Report.md)  
> **Original Implementation Plan:** [`nca-lm-implementation-plan.md`](file:///home/zenoguy/Desktop/projects/simulator/archive_nca_lm/nca-lm-implementation-plan.md)

---

## 1. Project Overview

This archive contains the complete experimental framework, model implementations, evaluation harnesses, unit test suites, and empirical artifacts investigating **Causal Neural Cellular Automata (NCA)** for autoregressive language modeling on WikiText-2 ($V=8192$ Byte-Level BPE, $10\text{M}$ parameter budget).

### Key Phases Executed:
- **Phase 0 (Floors):** Discrete 3-gram ($89.56$ Test PPL) and 5-gram ($99.40$ Test PPL) baselines.
- **Phase 1 (Baselines):** Full Causal Transformer ($42.30$ Test PPL), Sliding-Window Transformer ($42.30$ Test PPL), and GRU ($58.04$ Test PPL).
- **Phase 2 (2×2 Factorial Matrix):** Shared NCA vs. Unshared CNN controls ($3.6\text{M}$ and $10\text{M}$). Proved cellular weight sharing confers a significant channel-width capacity advantage ($105.92$ vs. $137.67$ PPL).
- **Phase 3 (Capability Probes):** Probed test-time compute scaling ($K$-scaling), impulse perturbation recovery, surface noise robustness, streaming complexity, and latent error contraction. Established that depth scaling fails outside training horizon ($K>6$ degrades) and noise robustness stems from convolutional locality.
- **Phase 4 (Pragmatic Hybrid):** Tested a $K=2$ pre-attention cellular stem on the Transformer ($+3.47\%$ params). Achieved **$41.11$ Test PPL** (beating pure Transformer $42.30$) and cut latent representation impulse damage by **$-55.9\%$** ($D=6.63$ vs. CNN's $14.98$ and Transformer's $15.03$).

---

## 2. Directory Structure

```
archive_nca_lm/
├── configs/                              # Model & training YAML configs (Levels 0, 1, 2, 4)
├── data/                                 # Tokenizer and dataset loaders
│   ├── dataset.py                        # Causal autoregressive dataset chunker
│   ├── tokenizer.py                      # Byte-Level BPE tokenizer wrapper
│   ├── tokenizer.json                    # Serialized 8,192-token BPE vocabulary
│   └── prepare_wikitext2.py              # WikiText-2 raw downloader
├── eval/                                 # Probing and evaluation modules
│   ├── perplexity.py                     # Standardized token-weighted perplexity
│   ├── probing_depth.py                  # Probe 3A: K-scaling sweep
│   ├── probing_perturbation.py           # Probe 3B/4B: Impulse perturbation tracking
│   ├── probing_robustness.py             # Probe 3C/4A: Surface token corruption sweep
│   ├── probing_streaming.py              # Probe 3D: State memory complexity
│   └── probing_denoising.py              # Probe 3E: Latent error contraction
├── models/                               # Neural architectures
│   ├── nca_lm.py                         # Causal NCA LM (shared & unshared rules)
│   ├── transformer_baseline.py           # Decoder-only Transformer (RoPE, SwiGLU)
│   ├── hybrid_transformer.py             # Phase 4 Hybrid NCA & CNN Control
│   ├── rnn_baseline.py                   # 3-layer Causal GRU baseline
│   ├── mamba_baseline.py                 # Reference Selective SSM
│   └── ngram.py                          # Kneser-Ney / smoothed n-gram baseline
├── notebooks/                            # Kaggle GPU execution runners
│   ├── kaggle_phase1_runner.ipynb
│   ├── kaggle_phase3_runner.ipynb
│   └── kaggle_phase4_runner.ipynb
├── outputs/                              # Evaluated JSON metrics across all phases
│   ├── level0/                           # N-gram metrics
│   ├── level1/                           # Baseline calibration tables & profiles
│   ├── level2/                           # 2×2 factorial matrix results
│   ├── level3/                           # Capability probe results & Gate 3 verdict
│   └── level4/                           # Hybrid evaluation & Gate 4 verdict
├── scripts/                              # Driver and evaluation scripts
│   ├── run_level0.py                     # N-gram runner
│   ├── run_level1.py                     # Baseline runner
│   ├── run_level2.py                     # Factorial matrix runner
│   ├── run_level3.py                     # Phase 3 capability hunt driver
│   ├── run_level4.py                     # Phase 4 hybrid evaluation driver
│   └── profile_models.py                 # FLOPs and parameter profiler
├── tests/                                # Automated unit test suite (32 tests)
│   ├── test_phase0.py
│   ├── test_phase1.py
│   ├── test_phase2.py
│   ├── test_phase3.py
│   └── test_phase4.py
├── train.py                              # Unified training entrypoint
├── requirements.txt                      # Project dependencies
├── conftest.py                           # Pytest configuration
├── nca-lm-implementation-plan.md         # Original roadmap
└── NCA_LM_Comprehensive_Research_Report.md # Full detailed research report
```

---

## 3. Running Unit Tests from the Archive

To run the test suite for this archived codebase:

```bash
pytest archive_nca_lm/tests/ -v
```
