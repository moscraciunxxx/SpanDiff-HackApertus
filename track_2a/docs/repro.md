# Reproducibility check, 13 Oct 2026

make predict reproduces committed files byte-for-byte on macOS 27.0.1 arm64 (Apple MPS, 48 GB RAM, Python 3.14.6, torch 2.14.1, bfloat16); cross-hardware drift noted as max |Δ| = 0.097791 on dev/full and 0.039058 on test between torch 2.13.0 and torch 2.14.1 MPS bfloat16 kernels (mean Spearman unchanged).

## Deterministic scoring settings

- Method: pre-registered variant (b) (`subword_first_l15`, `track_2a/docs/preregistration.md`) — `swiss-ai/Apertus-v1.5-8B` hidden state 15, paper-order subword-first `clamp(1 - max cosine, 0, 1)` followed by within-word mean-pooling (`track_2a/src/spandiff/score.py`).
- Batch size 1 with `padding=False` and `truncation=False` per 512-token window.
- Fixed seeds (`random.seed(0)`, `torch.manual_seed(0)`) and `torch.use_deterministic_algorithms(True, warn_only=True)`.
- Transformers: `github.com/swiss-ai/transformers` at `3797303dda74844e3d1f8977ff5518bb91f818b4` (`5.14.0.dev0`), `tokenizers==0.22.2`, `numpy==2.3.5`, `scipy==1.16.3`, `accelerate==1.15.0`.

## Two-run scratch verification and committed file check

Running the deterministic `make predict` scorer twice into clean scratch directories (`/tmp/spandiff-win-1` and `/tmp/spandiff-win-2`), scoring the 56-document test inputs twice, and assembling the 224-document full-set files produced identical bytes across both runs (`max |Δ| = 0.0`) and matched all nine committed files (`track_2a/predictions/{dev,test,full}/SpanDiff_admin_{de,fr,it}.jsonl.jsonl`) byte-for-byte:

| Split | Docs / lang | Forward-pass tokens | Score seconds | SHA-256 (de / fr / it) |
| --- | --- | --- | --- | --- |
| `dev` | 168 | 564,349 | 549.8 | `cf2964000419...` / `ef2f1fa6293b...` / `3a170ac7d1a1...` |
| `test` | 56 | 207,075 | 354.1 (run 1), 261.8 (run 2) | `36a9b37f0127...` / `904ec060e458...` / `8ffd5a1606df...` |
| `full` | 224 | 771,424 | assembled from `dev` + `test` (`0..223`) | `a5b475b13928...` / `330425ef4fc6...` / `9c2ecb965e2e...` |

Word-mean-pool hidden state 15 (`0,296 / 0,191 / 0,324`, mean `0.2703`) is retained in `track_2a/predictions/l15` (`bb0c1cac606d...` / `56b58418d67c...` / `c83223ce1293...`) and also verified byte-identical across two deterministic runs.

## Official development regrade (`--split dev` only)

- Submitted (`predictions/dev`, subword-first L15): Spearman `de 0,309 ; fr 0,199 ; it 0,333` (unweighted mean `0.2803`), Kendall's Tau-b `de 0,249 ; fr 0,154->0,161 ; it 0,266` (`0,249 / 0,161 / 0,266`).
- Word-mean-pool L15 ablation (`predictions/l15`): Spearman `de 0,296 ; fr 0,191 ; it 0,324` (unweighted mean `0.2703`), Kendall's Tau-b `de 0,238 ; fr 0,154 ; it 0,259`.
- Test gold was never opened or graded.
