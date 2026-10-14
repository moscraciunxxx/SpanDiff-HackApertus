# Pre-registered representation study, 13–14 Oct 2026

Committed before running candidate variants. All selection uses the development split only (`dev/train` 134 docs/lang and `dev/val` 34 docs/lang); the sealed evaluation split and its labels are never read or scored during selection.

## Batch 1 (13 Oct 2026, completed on branch `improve`)

- Baseline (`L15` word mean-pool): `swiss-ai/Apertus-v1.5-8B` hidden state 15, subwords mean-pooled into whitespace tokens, then `clamp(1 - max cosine, 0, 1)`. Official `--split dev` Spearman: `de 0,296 ; fr 0,191 ; it 0,324`, mean `0.2703`.
- Variants tested:
  1. Variant (a) `mean_l12_17` (word mean-pool): `0,285 / 0,183 / 0,316`, mean `0.2613` — **LOST**.
  2. Variant (b) `subword_first_l15` (paper-order subword-first at hidden state 15): `0,309 / 0,199 / 0,333`, mean `0.2803` (`dev/train` `0.2843` vs `0.2734`, `dev/val` `0.2592` vs `0.2527`, full-dev `+0.0100 >= +0.0050`) — **WON & Adopted**.
  3. Variant (c) `combined_l12_17_subword_first`: `0,298 / 0,192 / 0,325`, mean `0.2717` — **LOST**.

---

## Batch 2 — Tier 1 Post-Processing Sprint on Saved Hidden States (completed on branch `sprint`)

- **Baseline to beat:** `subword_first_l15` (official `--split dev` Spearman `de 0,309 ; fr 0,199 ; it 0,333`, unweighted mean `0.2803`; `dev/train` `0.2843`, `dev/val` `0.2592`).
- **Selection & adoption gate:** Select parameters on `dev/train` IDs (134/lang), confirm on `dev/val` IDs (34/lang), and adopt if and only if it beats the current best on **both** `dev/train` and `dev/val` **and** improves the full-development official `--split dev` printed mean Spearman by **at least `+0.0030`**.

### Pre-registered Tier-1 items and recorded outcomes

1. **Item 1 — Bidirectional max / CSLS alignment (`alpha=0.50`, `k=5`):** `dev/train` `0.2909`, `dev/val` `0.2659`, official `--split dev` printed mean `0.2870` (`0,320 / 0,200 / 0,341`, `+0.0067 >= +0.0030`) — **WON**.
2. **Item 2 — Span smoothing (`w=2` triangular-weighted over non-punctuation tokens):** `dev/train` `0.3263`, `dev/val` `0.2973`, official `--split dev` printed mean `0.3217` (`0,353 / 0,227 / 0,385`, `+0.0414 >= +0.0030`) — **WON**.
3. **Item 3 — Per-language / per-document calibration (`mean_sub_joint alpha=0.20`, `zscore`, `quantile`):** Best `dev/train` `0.2836`, `dev/val` `0.2586`, printed mean `0.2797` (`-0.0006`) — **LOST**.
4. **Item 4 — Function-word / inverse token frequency weighting (`idf power=-0.05`):** `dev/train` `0.2848`, `dev/val` `0.2598`, printed mean `0.2810` (`+0.0007 < +0.0030`) — **LOST**.
5. **Item 5 — Number / date / named-entity handling (`match=1.0, miss=1.2, cap=1.0`):** `dev/train` `0.2861`, `dev/val` `0.2603`, printed mean `0.2820` (`+0.0017 < +0.0030`) — **LOST**.
6. **Item 6 — Top-k soft alignment (`k=5, T=0.02`):** `dev/train` `0.2849`, `dev/val` `0.2601`, printed mean `0.2813` (`+0.0010 < +0.0030`) — **LOST**.
7. **Item 7 — Sentence-constrained alignment (`lam=0.50`, matched sentence `+/- 1`):** `dev/train` `0.2925`, `dev/val` `0.2651`, printed mean `0.2880` (`0,320 / 0,205 / 0,339`, `+0.0077 >= +0.0030`) — **WON**.
8. **Item 8 — Learned layer weights (layers 10–20, `0.168*L12 + 0.340*L14 + 0.492*L15`):** `dev/train` `0.2854`, `dev/val` `0.2597`, printed mean `0.2813` (`+0.0010 < +0.0030`) — **LOST**.
9. **Item 9 — Contrastive centering (`doc_mean beta=0.50` / `global_mean beta=1.00`):** `doc_mean` `dev/train` `0.2887`, `dev/val` `0.2647`, printed mean `0.2850` (`+0.0047`); `global_mean` `dev/train` `0.2902`, `dev/val` `0.2692`, printed mean `0.2870` (`+0.0067 >= +0.0030`) — **WON**.
10. **Item 10 — Whitening (`abtt1` top-1 PC removal vs `pca256` whitening):** `abtt1` `dev/train` `0.2904`, `dev/val` `0.2689`, printed mean `0.2873` (`+0.0070 >= +0.0030`) — **WON** (`pca256` printed mean `0.2770` lost).
11. **Tier-1 Combination (Item 2 `w=2` triangular span smoothing + Item 9 document mean centering `beta=0.50` + Item 1 bidirectional CSLS `alpha=0.50` + Item 7 sentence-constrained alignment `lam=0.35`):**
    - `dev/train` mean Spearman: `0.3383` (vs `0.2843`, `+0.0540`)
    - `dev/val` mean Spearman: `0.3115` (vs `0.2592`, `+0.0523`)
    - Full-dev official `--split dev` (`docs/official_dev_stdout_tier1_combo.txt`): Spearman `de 0,373 ; fr 0,231 ; it 0,398`, unweighted mean **`0.3340`** (`+0.0537 >= +0.0030`), Kendall `de 0,302 ; fr 0,187 ; it 0,320`. **Adopted as current best (`0.3340`).**

---

## Batch 3 — Tier 2 New Apertus Forward-Pass Variants (completed on branch `sprint`)

- **Baseline to beat:** Tier-1 Combination (`dev/train` `0.3383`, `dev/val` `0.3115`, official `--split dev` printed mean `0.3340`, adoption gate `>= 0.3370`).
- **Recorded outcomes:**
  11. **Item 11 — Translation-style prompt prefix (`"Translate the following text:\n"`):** Raw subword-first `dev/train` `0.2886`, `dev/val` `0.2604`, printed mean `0.2843`; with Tier-1 Combination `dev/train` `0.3402`, `dev/val` `0.3128`, printed mean `0.3357` (`0,373 / 0,232 / 0,402`, `+0.0017 < +0.0030` gate) — **LOST (below +0.0030 gate)**.
  12. **Item 12 — Paired context (`"text_a \n\n text_b"` in one pass):** Raw `dev/train` `0.2512`, `dev/val` `0.2249`, printed mean `0.2467`; with Tier-1 Combination `dev/train` `0.3009`, `dev/val` `0.2704`, printed mean `0.2960` (`-0.0380`) — **LOST**.
  13. **Item 13 — Shorter-window overlapping encoding (256-token windows, stride 128):** Raw `dev/train` `0.2715`, `dev/val` `0.2560`, printed mean `0.2690`; with Tier-1 Combination `dev/train` `0.3332`, `dev/val` `0.3132`, printed mean `0.3303` (`-0.0037`) — **LOST**.
  14. **Item 14 — Apertus layer-15 cross-side attention mass feature:** Attention alone `dev/train` `-0.0251`, `dev/val` `-0.0184`, printed mean `-0.0237`; 50/50 rank average with Tier-1 Combination printed mean `0.1753`; 80/20 rank blend printed mean `0.2630` (`-0.0710`) — **LOST**.
- **Final adopted method:** Tier-1 Combination on `swiss-ai/Apertus-v1.5-8B` hidden state 15 (`0.3340` mean Spearman).
