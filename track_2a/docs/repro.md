# Reproducibility check, 14 Oct 2026

make predict reproduces committed files byte-for-byte on macOS 27.0.1 arm64 (Apple MPS, 48 GB RAM, Python 3.14.6, torch 2.14.1, bfloat16); cross-hardware drift noted as max |Δ| = 0.097791 on dev/full and 0.039058 on test between torch 2.13.0 and torch 2.14.1 MPS bfloat16 kernels.

## Deterministic scoring settings

- Method: pre-registered Tier-1 Combination (`track_2a/docs/preregistration.md`) on `swiss-ai/Apertus-v1.5-8B` hidden state 15 — document-mean contrastive centering (`beta = 0.50`), bidirectional CSLS hubness margin (`alpha = 0.50, k = 5`), sentence-constrained alignment (`lambda = 0.35`, matched sentence `+/- 1`), paper-order subword-first `clamp(1 - max cosine, 0, 1)` pooled into whitespace tokens, and window-2 triangular span smoothing over content tokens (`track_2a/src/spandiff/score.py`).
- Batch size 1 with `padding=False` and `truncation=False` per 512-token window.
- Fixed seeds (`random.seed(0)`, `torch.manual_seed(0)`) and `torch.use_deterministic_algorithms(True, warn_only=True)`.
- Transformers: `github.com/swiss-ai/transformers` at `3797303dda74844e3d1f8977ff5518bb91f818b4` (`5.14.0.dev0`), `tokenizers==0.22.2`, `numpy==2.3.5`, `scipy==1.16.3`, `accelerate==1.15.0`.

## Two-run scratch verification and committed file check

Running the deterministic `make predict` scorer twice into clean scratch directories (`/tmp/spandiff-sprint-win-1` and `/tmp/spandiff-sprint-win-2`), scoring the 56-document test inputs twice, and assembling the 224-document full-set files produced identical bytes across both runs (`max |Δ| = 0.0`) and matched all nine committed files (`track_2a/predictions/{dev,test,full}/SpanDiff_admin_{de,fr,it}.jsonl.jsonl`) byte-for-byte:

| Split | Docs / lang | Forward-pass tokens | Score seconds | SHA-256 (de / fr / it) |
| --- | --- | --- | --- | --- |
| `dev` | 168 | 564,349 | 280.2 | `a34b70adf48e...` / `199bc15eadb5...` / `446efc6e5cf6...` |
| `test` | 56 | 207,075 | 165.8 (run 1), 182.1 (run 2) | `b468ec41702c...` / `52201cfed167...` / `010e4665f17d...` |
| `full` | 224 | 771,424 | assembled from `dev` + `test` (`0..223`) | `187bbcfbf276...` / `104592d5dafa...` / `e914375cc35e...` |

Word-mean-pool hidden state 15 (`0,296 / 0,191 / 0,324`, mean `0.2703`) is retained in `track_2a/predictions/l15` (`bb0c1cac606d...` / `56b58418d67c...` / `c83223ce1293...`).

## Official development regrade (`--split dev` only)

- Submitted (`predictions/dev`, L15 Tier-1 Combination): Spearman `de 0,373 ; fr 0,231 ; it 0,398` (unweighted mean `0.3340`), Kendall's Tau-b `de 0,302 ; fr 0,187 ; it 0,320`.
- Unsmoothed subword-first L15 ablation: Spearman `de 0,309 ; fr 0,199 ; it 0,333` (unweighted mean `0.2803`), Kendall's Tau-b `de 0,249 ; fr 0,161 ; it 0,266`.
- Word-mean-pool L15 ablation (`predictions/l15`): Spearman `de 0,296 ; fr 0,191 ; it 0,324` (unweighted mean `0.2703`), Kendall's Tau-b `de 0,238 ; fr 0,154 ; it 0,259`.
- Test gold was never opened or graded.

## Public-text reproduction (`make predict-test`)

On 14 Oct 2026, `make predict-test` downloaded public SwissGov-RSD text fields only, dropped label fields, scored hidden state 15 with the shipped constants, and wrote scratch copies of the test and full predictions. All six files matched the committed bytes (`full` and `test`, de/fr/it). Forward-pass tokens on that run: 771,424. Seconds: 383.5. Scratch directory: `/tmp/spandiff-predict-test`. The committed files were not replaced and were not graded.
