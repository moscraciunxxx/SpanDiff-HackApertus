# Pre-registered representation study, 13 Oct 2026

Committed on branch `improve` before running any candidate variant. All selection uses the development split only; the sealed evaluation split and its labels are never read or scored during selection.

## Baseline to beat

- Current submitted method (`L15`): `swiss-ai/Apertus-v1.5-8B` hidden state 15, subwords mean-pooled into whitespace tokens, then `clamp(1 - max cosine, 0, 1)`.
- Official development print (`--split dev`, 168 documents per language): Spearman `de 0,296 ; fr 0,191 ; it 0,324`, unweighted mean `0.2703`.

## Pre-registered variants (all on `swiss-ai/Apertus-v1.5-8B`, same deterministic setup)

All three variants use the identical deterministic forward pass: 512-token windows (`padding=False`, `truncation=False`, batch size 1), `random.seed(0)`, `torch.manual_seed(0)`, `torch.use_deterministic_algorithms(True, warn_only=True)`, `bfloat16` on Apple MPS with `torch 2.14.1` and `github.com/swiss-ai/transformers` at `3797303dda74844e3d1f8977ff5518bb91f818b4`.

1. **Variant (a) — `mean_l12_17` (mean of hidden states 12–17):**
   For each 512-token window, average `hidden_states[12]` through `hidden_states[17]` elementwise (`sum(hidden_states[12:18]) / 6.0`), then apply the standard subword-to-word mean-pooling and word-level `clamp(1 - max cosine, 0, 1)`.
2. **Variant (b) — `subword_first_l15` (paper order at hidden state 15):**
   Read `hidden_states[15]`. Stitch window subword vectors and offsets across the document side. Compute `clamp(1 - max cosine, 0, 1)` for every subword against all content subwords on the other side, then average the clamped subword difference scores inside each whitespace token (`1.0` when a whitespace token has no overlapping subword).
3. **Variant (c) — `combined_l12_17_subword_first` ((a) + (b) combined):**
   Average `hidden_states[12]` through `hidden_states[17]` elementwise per window, then score with the paper-order subword-first `clamp(1 - max cosine, 0, 1)` followed by within-word averaging.

## Selection and adoption rule

- Partition the 168 development documents per language into the published upstream `dev/train` IDs (134 per language) and `dev/val` IDs (34 per language).
- Pick on `dev/train` IDs (134/lang) and confirm on `dev/val` IDs (34/lang) using average-rank token-level Spearman (dropping gold `-1` positions, macro-averaged across `de`, `fr`, `it`).
- Also run the official evaluator (`python -m scripts.evaluate_predictions_admin ... --split dev`) on all 168 development documents per language and save the exact stdout logs under `track_2a/docs/`:
  - `track_2a/docs/official_dev_stdout_l12_17.txt`
  - `track_2a/docs/official_dev_stdout_subword_l15.txt`
  - `track_2a/docs/official_dev_stdout_combined.txt`
- **Adoption gate:** Adopt a candidate variant if and only if it beats `L15` on **both** `dev/train` mean Spearman and `dev/val` mean Spearman **and** improves the full-development official `--split dev` printed mean Spearman by **at least `+0.0050`** (`>= 0.2753` vs `0.2703`). Otherwise keep `L15` (`0.2703`) as the primary submission and report all three variants in the technical report ablation table.
