# Simulator

## Project Archives

This repository preserves two completed research investigations:

### 1. Causal Neural Cellular Automata for Language Modeling (`archive_nca_lm/`)
- **Scope:** Systematic investigation of Neural Cellular Automata (NCA) across five phases (Phases 0–4) on WikiText-2 ($10\text{M}$ parameter budget).
- **Key Findings:**
  - Characterized depth scaling failure outside training horizon ($K>6$ degrades) and identified parameter concentration into channel width as the primary benefit of cellular weight sharing ($105.9$ vs $137.7$ PPL).
  - Designed and evaluated the **Pragmatic Hybrid NCA-Transformer** ($K=2$ cellular stem, $+3.47\%$ params), achieving **$41.11$ Test PPL** (outperforming the $42.30$ pure Transformer baseline) and cutting internal representation impulse damage by **$-55.9\%$** ($D=6.63$ vs. CNN's $14.98$ and Transformer's $15.03$).
- **Primary Report:** [`archive_nca_lm/NCA_LM_Comprehensive_Research_Report.md`](file:///home/zenoguy/Desktop/projects/simulator/archive_nca_lm/NCA_LM_Comprehensive_Research_Report.md)
- **Directory:** [`archive_nca_lm/`](file:///home/zenoguy/Desktop/projects/simulator/archive_nca_lm)

### 2. Memory-Augmented NCA on 1D KdV Soliton Dynamics (`archive/`)
- **Scope:** Investigation of memory-augmented Neural Cellular Automata on non-linear dispersive wave equations (Korteweg–de Vries soliton propagation).
- **Documentation:** [`archive/README.md`](file:///home/zenoguy/Desktop/projects/simulator/archive/README.md)
- **Directory:** [`archive/`](file:///home/zenoguy/Desktop/projects/simulator/archive)

---

## Active Workspace

> **Notice:** The repository is currently clean and primed for the next research pivot.

