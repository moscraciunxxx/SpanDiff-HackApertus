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

---

## Batch 6 — Sprint 3 (pre-registered on `sprint3` before running)

Committed before any Sprint 3 forward pass. Development documents only. The sealed split is not read or scored. Ids: `track_2a/docs/sprint2_dev_ids.json`. Folds: numeric id modulo 5. Metric: token-pooled Spearman, gold `-1` dropped, `labels_a` and `labels_b` concatenated, then the unweighted mean of de, fr, and it.

- **Gate:** `dev/train >= 0.3433` and `dev/val >= 0.3165` and at least 4 of 5 folds strictly above the frozen Tier-1 scores.
- **Superseded before any forward pass:** diagonal-0.25 plus window-3 smoothing; equal average of layers 14, 15, and 16; context prefix plus diagonal 0.25. Replaced by the tracks below. They were not run.

### Track A — new Apertus signals, training-free

1. **Translation surprisal.** Prompt with source: `Translate English to {German|French|Italian}:\n` + side A + `\n` + side B. Prompt without source: the same prefix plus side B only. The swap uses `Translate {German|French|Italian} to English:\n`. Each target side is scored in 512-token windows, including the leading BOS. The target chunk is never truncated. Source tokens that do not fit are dropped from the left. Token surprisal is the mean subword `(logp without source) - (logp with source)`. Rank-average with the frozen Tier-1 score (`0.5` each after `rank/(n-1)`).
2. **Bidirectional surprisal.** Idea 1 in both directions. Map the other direction onto tokens through the greedy sentence match (best sentence, `+/- 0`). Rank-average both surprisals with the frozen score, equal thirds.
- **Stack rule, locked before surprisal is scored.** If idea 1 or idea 2 passes the gate alone, also add the fixed mismatch bonus of `0.15` to that variant and clip to `[0, 1]`. The bonus weight is not retuned. The stack is adopted only if it passes the gate itself.
- **One-direction variant, locked before it is scored.** Side B only: rank-average the frozen score with the English-to-target surprisal. Side A stays the frozen Tier-1 score. No swap. The same stack rule applies if this variant passes alone.
3. **Attention alignment.** One concatenated pass, `side A + "\n" + side B`, hidden state 15. Token score is `1 - max` attention mass from that token onto the other side. If SDPA returns no weights, this idea is recorded as not runnable and is not replaced by cosine.
4. **Multi-layer CSLS.** Full Tier-1 pipeline at hidden states 12, 15, and 18. Rank-average the three smoothed score vectors. Equal weights. No other layers.
5. **Drop test, one extra forward per document.** Remove the single best-matching source sentence for every target sentence (union of those spans), re-encode, and take one minus the cosine of the original and ablated target-sentence means. Spread that value to the sentence's tokens. Rank-average with the frozen score.

### Track B — structure-aware post-processing, constants locked

6. **Mismatch bonus `0.15`.** From each token, extract digits of percents, `CHF` amounts, numeric dates, `Art.` / `Abs.` / `al.` references, and other numbers. If any digit string is absent from the other side, add `0.15` and clip to `[0, 1]`.
7. **Sentence prior.** Sentence vector is the mean of centered hidden-state-15 subwords. Multiply the frozen token score by `1 - max` cosine to the other side's sentences, then clip to `[0, 1]`.
8. **Function-word damping `0.5`.** Punctuation and these stopwords are multiplied by `0.5`: English `the a an of and to in for on with at by from as or`; German `der die das den dem des ein eine und zu von mit auf für im in ist`; French `le la les de des du un une et à au aux en dans pour par sur est`; Italian `il lo la i gli le di del della dei e a in per con su è`.
9. **Length-adaptive smoothing.** If either side has at least 800 whitespace tokens, apply one extra window-2 triangular smooth to the frozen scores. Shorter documents stay unchanged. The trigger does not use gold.

### Recorded after the first four

- **6 mismatch bonus:** train `0.3418`, val `0.3165`, folds `5/5` — **LOST** (train below `0.3433`).
- **7 sentence prior:** train `0.3004`, val `0.2856`, folds `0/5` — **LOST**.
- **8 function-word damping:** train `0.2378`, val `0.2143`, folds `0/5` — **LOST**.
- **9 length-adaptive smoothing:** train `0.3398`, val `0.3129`, folds `5/5` — **LOST**.
- **4 multi-layer rank average (L12, L15, L18):** train `0.2814`, val `0.2644`, folds `0/5` — **LOST**.
- **1 translation surprisal:** train `0.2106`, val `0.2018`, folds `0/5` — **LOST**. The mismatch stack was not run, because this variant did not pass alone.
- **1 one-direction:** train `0.2243`, val `0.2173`, folds `0/5` — **LOST**. Side A stayed frozen. The mismatch stack was not run.
- **10 ridge stacker, out of fold:** train `0.3458`, val `0.3277`, folds `5/5` — **PASSES** the development gate. Per-language out-of-fold Spearman de `0.3932`, fr `0.2392`, it `0.3971` against `0.373 / 0.231 / 0.398`. Italian drops `0.0009`, inside `0.01`. Alpha stayed `1.0`. Four near-zero weights flip sign across folds (relative position, punctuation, token length, side); each stays below `0.003` and is flagged. The graded development files are these out-of-fold scores. Test and full use the ridge fit on all development documents.
- **2 bidirectional surprisal:** train `0.1786`, val `0.1722`, folds `0/5` — **LOST**. The mismatch stack was not run.
- **3 attention alignment:** not runnable. SDPA returned no attention weights (`output_attentions` is empty). Cosine was not substituted.

### Track C — light learning

10. **Ridge stacker, `alpha = 1.0`.** At most these features: frozen score, surprisal rank, mismatch indicator, sentence prior, relative position, stopword, punctuation, token character length, document length ratio, side. Standardize inside each training fold. Doc folds are id modulo 5. Only out-of-fold scores are gated. A stack that adds a feature must pass the gate again.
11. **Qwen silver, dev comparison only.** Run the Tier-1 pipeline on `Qwen/Qwen2.5-VL-7B-Instruct` over public non-development texts. Do not read sealed labels. Fit idea 10 on those silver scores, then apply it to Apertus features. Submitted numbers still come only from Apertus features.

### Track D — robustness, done whether or not a scorer is adopted

12. **Precision.** Compare the final pipeline in float32 and bfloat16. If float32 does not fit, score the first 10 development ids per language and report that subset.
13. **Scatter.** Per-document Spearman of the submitted scores against the final-layer baseline.
14. **Failure gallery.** The five development documents with the lowest per-document Spearman, with a short reason each.
