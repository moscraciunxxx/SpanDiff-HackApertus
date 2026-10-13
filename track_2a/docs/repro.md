# Reproducibility check, 13 Oct 2026

make predict reproduces committed files byte-for-byte on macOS 27.0.1 arm64 (Apple MPS, 48 GB RAM, Python 3.14.6, torch 2.14.1, bfloat16); cross-hardware drift noted as max |Δ| = 0.097791 on dev/full and 0.039058 on test under torch 2.13.0 (Python 3.13.9, mean Spearman 0.2703 unchanged: de 0,295 ; fr 0,191 ; it 0,325).

## Deterministic scoring settings

- Batch size 1 with `padding=False` and `truncation=False` per 512-token window (`track_2a/src/spandiff/score.py`).
- Fixed seeds (`random.seed(0)`, `torch.manual_seed(0)`) and `torch.use_deterministic_algorithms(True, warn_only=True)`.
- Transformers: `github.com/swiss-ai/transformers` at `3797303dda74844e3d1f8977ff5518bb91f818b4` (`5.14.0.dev0`), `tokenizers==0.22.2`, `numpy==2.3.5`, `scipy==1.16.3`, `accelerate==1.15.0`.

## Two-run scratch verification and committed file check

Running `make predict` twice into two clean scratch directories (`/tmp/spandiff-det-1/dev` and `/tmp/spandiff-det-2/dev`), scoring the 56-document test inputs, and assembling the 224-document full-set files produced identical bytes across both runs (`max |Δ| = 0.0`) and matched all nine committed files (`track_2a/predictions/{dev,test,full}/SpanDiff_admin_{de,fr,it}.jsonl.jsonl`) byte-for-byte:

| Split | Docs / lang | Forward-pass tokens | Score seconds | SHA-256 (de / fr / it) |
| --- | --- | --- | --- | --- |
| `dev` | 168 | 564,349 | 649.6 (run 1), 751.7 (run 2) | `bb0c1cac606d...` / `56b58418d67c...` / `c83223ce1293...` |
| `test` | 56 | 207,075 | 258.7 | `8b47b997ba18...` / `eb4c5e63d3d1...` / `0e6c86a965e4...` |
| `full` | 224 | 771,424 | assembled from `dev` + `test` (`0..223`) | `d1845f92fc42...` / `93d1b1004819...` / `7e024972265f...` |

## Official development regrade (`--split dev` only)

- Spearman: `de 0,296 ; fr 0,191 ; it 0,324` (unweighted mean `0.2703`).
- Kendall's Tau-b: `de 0,238 ; fr 0,154 ; it 0,259`.
- Test gold was never opened or graded.
