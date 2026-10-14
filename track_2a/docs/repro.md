# Reproducibility check, 14 Oct 2026

On 14 Oct 2026, `make predict` run twice produced byte-identical `predictions/rerun` files, and `make predict-test` matched the committed test and full files. Machine: macOS 27.0.1 arm64, Apple MPS, 48 GB RAM, Python 3.14.6, torch 2.14.1, bfloat16. An earlier Tier-1 cross-hardware check had max absolute drift 0.097791 on dev/full and 0.039058 on test between torch 2.13.0 and torch 2.14.1.

## Deterministic scoring settings

- Method: pre-registered Tier-1 Combination (`track_2a/docs/preregistration.md`) on `swiss-ai/Apertus-v1.5-8B` hidden state 15 — document-mean contrastive centering (`beta = 0.50`), bidirectional CSLS hubness margin (`alpha = 0.50, k = 5`), sentence-constrained alignment (`lambda = 0.35`, matched sentence `+/- 1`), paper-order subword-first `clamp(1 - max cosine, 0, 1)` pooled into whitespace tokens, and window-2 triangular span smoothing over content tokens (`track_2a/src/spandiff/score.py`).
- Batch size 1 with `padding=False` and `truncation=False` per 512-token window.
- Fixed seeds (`random.seed(0)`, `torch.manual_seed(0)`) and `torch.use_deterministic_algorithms(True, warn_only=True)`.
- Transformers: `github.com/swiss-ai/transformers` at `3797303dda74844e3d1f8977ff5518bb91f818b4` (`5.14.0.dev0`), `tokenizers==0.22.2`, `numpy==2.3.5`, `scipy==1.16.3`, `accelerate==1.15.0`.

## Two-run scratch verification and committed file check

The graded development files are out-of-fold ridge scores. `make predict` applies `docs/ridge_weights.json` and does not refit. Two runs on 14 Oct 2026 wrote identical bytes to `predictions/rerun`. That rerun is the full-development fit, so it matches the development ids inside `predictions/full` and it is not the graded out-of-fold file. `make predict-test` rebuilt test and full from public text with a fresh feature cache and matched those six committed files byte for byte.

| Split | Docs / lang | What it is | SHA-256 (de / fr / it) |
| --- | --- | --- | --- |
| `dev` | 168 | out-of-fold ridge | `1098404f7ad7...` / `4672894bd3bb...` / `a689c9367e1e...` |
| `test` | 56 | ridge fit on all development documents | `b987b965792d...` / `60101727b952...` / `1be0108f7892...` |
| `full` | 224 | same fit, ids `0..223` | `71379467445d...` / `95cf73fdddfd...` / `bb324dd2aee1...` |

Word-mean-pool hidden state 15 (`0,296 / 0,191 / 0,324`, mean `0.2703`) is retained in `track_2a/predictions/l15` (`bb0c1cac606d...` / `56b58418d67c...` / `c83223ce1293...`).

## Official development regrade (`--split dev` only)

- Submitted (`predictions/dev`, out-of-fold ridge): Spearman `de 0,393 ; fr 0,239 ; it 0,397` (unweighted mean `0.3430`), Kendall's Tau-b `de 0,324 ; fr 0,196 ; it 0,322`.
- Tier-1 Combination feature: Spearman `de 0,373 ; fr 0,231 ; it 0,398` (unweighted mean `0.3340`), Kendall's Tau-b `de 0,302 ; fr 0,187 ; it 0,320`.
- Unsmoothed subword-first L15 ablation: Spearman `de 0,309 ; fr 0,199 ; it 0,333` (unweighted mean `0.2803`), Kendall's Tau-b `de 0,249 ; fr 0,161 ; it 0,266`.
- Word-mean-pool L15 ablation (`predictions/l15`): Spearman `de 0,296 ; fr 0,191 ; it 0,324` (unweighted mean `0.2703`), Kendall's Tau-b `de 0,238 ; fr 0,154 ; it 0,259`.
- Test gold was never opened or graded.

## Public-text reproduction (`make predict-test`)

On 14 Oct 2026, `make predict-test` downloaded public SwissGov-RSD text fields only, dropped label fields, applied `docs/ridge_weights.json` without refitting, and wrote scratch copies of the test and full predictions. All six files matched the committed bytes (`full` and `test`, de/fr/it). Scratch directory: `/tmp/spandiff-predict-test`. The committed files were not replaced and were not graded.
