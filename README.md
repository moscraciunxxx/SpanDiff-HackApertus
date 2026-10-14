# SpanDiff

Repository: https://github.com/moscraciunxxx/SpanDiff-HackApertus

SpanDiff scores token-level semantic drift on Swiss federal pages for Track 2A (UZH). It reads hidden state 15 of `swiss-ai/Apertus-v1.5-8B` and applies the pre-registered Tier-1 Combination (`track_2a/docs/preregistration.md`): document-mean contrastive centering (`beta = 0.50`), bidirectional CSLS hubness margin (`alpha = 0.50, k = 5`), sentence-constrained alignment (`lambda = 0.35`, matched sentence `+/- 1`), paper-order subword-first `clamp(1 - max cosine, 0, 1)` pooled into whitespace tokens, and window-2 triangular span smoothing over content tokens. No text is generated. Apertus 70B was not run.

## Results

Development split, official script, `--split dev`. The mean is the unweighted average of the three printed Spearmans, to 4 decimals.

| Setup (on `swiss-ai/Apertus-v1.5-8B` unless noted) | de | fr | it | mean |
| --- | --- | --- | --- | --- |
| Paper DiffAlign recipe on Apertus 8B, final layer, clamped & unclamped | 0.053 | 0.056 | 0.106 | 0.0717 |
| SpanDiff ablation, hidden state 8 (`track_2a/predictions/ablation_l8`, word mean-pool) | 0.245 | 0.165 | 0.282 | 0.2307 |
| Pre-registered (a): mean of hidden states 12–17 (word mean-pool) | 0.285 | 0.183 | 0.316 | 0.2613 |
| SpanDiff ablation, hidden state 15 (`track_2a/predictions/l15`, word mean-pool) | 0.296 | 0.191 | 0.324 | 0.2703 |
| Pre-registered (c): hidden states 12–17 + subword-first | 0.298 | 0.192 | 0.325 | 0.2717 |
| Pre-registered (b): hidden state 15, paper-order subword-first ablation | 0.309 | 0.199 | 0.333 | 0.2803 |
| Open-weight dev comparison: `Qwen/Qwen2.5-VL-7B-Instruct` (layer 14/28, same Tier-1 Combo) | 0.351 | 0.217 | 0.381 | 0.3163 |
| Hidden state 15, Tier-1 Combination (frozen feature inside the ridge) | 0.373 | 0.231 | 0.398 | 0.3340 |
| **SpanDiff submitted, out-of-fold ridge (`track_2a/predictions/dev`)** | **0.393** | **0.239** | **0.397** | **0.3430** |

Development scores are out-of-fold. Each development document was scored by a ridge fit on the other documents, with folds equal to the numeric id modulo 5. No document is in both sides of its fold. Test predictions, 56 documents per language, and full-set predictions, 224 documents per language, apply one ridge fit on all development documents. The coefficients are `track_2a/docs/ridge_weights.json`. They are included for the organizers to score. We never graded the test split. `make run` grades `track_2a/predictions/dev` only. `track_2a/predictions/l15` retains the word-mean-pool hidden-state-15 ablation (`0.2703`). Printed development form of the submission, decimal commas: de 0,393 ; fr 0,239 ; it 0,397. Kendall de 0,324 ; fr 0,196 ; it 0,322. The Tier-1 Combination print remains de 0,373 ; fr 0,231 ; it 0,398, mean 0.3340, Kendall de 0,302 ; fr 0,187 ; it 0,320 (`track_2a/docs/official_dev_stdout_tier1_combo.txt`). The official script does not print that mean. `make run` prints it after the table. The ablation and comparison rows are in `track_2a/docs/official_dev_stdout_subword_l15.txt`, `track_2a/docs/official_dev_stdout_L15.txt`, `track_2a/docs/official_dev_stdout_l12_17.txt`, `track_2a/docs/official_dev_stdout_combined.txt`, `track_2a/docs/official_dev_stdout_qwen_comparison.txt`, `track_2a/docs/official_dev_stdout.txt`, and `track_2a/docs/official_dev_stdout_last.txt`. Static visual demo: `track_2a/docs/demo.html`.

## Efficiency

Development forward pass: 564,349 tokens, 0 generated. Layer-15 Tier-1 Combination scoring took 280.2 seconds on an Apple GPU after the weights were cached.

## Run

Judges run this from the repository root. It builds the Docker image and grades `track_2a/predictions/dev` with `--split dev`. It does not load the model.

```bash
make run
```

`make run-local` is that same grade when Docker is not installed. Install the grade dependencies with `pip install -r track_2a/requirements-grade.txt` (the existing pins, plus torch and transformers). `LLM_NAME`, `LLM_BASE_URL`, and `LLM_API_KEY` are not the scorer. Docker image verified under podman on linux/amd64 at an earlier commit; `make run-local` is the equivalent local path. Windows Docker Desktop instructions are in `track_2a/docs/DOCKER_CHECK.md`.

`make predict` loads `track_2a/docs/ridge_weights.json` and writes that full-development fit to `track_2a/predictions/rerun`. It does not refit, and it does not replace `track_2a/predictions/dev`. The graded development files stay the out-of-fold scores. It needs a Hugging Face token only if the weights are not cached, and about 16 GB.

```bash
make predict
```

## Reproduce

`make predict-test` rebuilds the test and full-set predictions in a scratch directory from public text fields only, applies the saved ridge, then byte-compares them to the committed files. It does not replace those files and it does not grade them. Test texts are `https://huggingface.co/datasets/ZurichNLP/SwissGov-RSD/resolve/main/data/de/test_de.jsonl`, `https://huggingface.co/datasets/ZurichNLP/SwissGov-RSD/resolve/main/data/fr/test_fr.jsonl`, and `https://huggingface.co/datasets/ZurichNLP/SwissGov-RSD/resolve/main/data/it/test_it.jsonl`. The Hub has no separate full file. Full-set texts are those three files plus the public development texts `https://huggingface.co/datasets/ZurichNLP/SwissGov-RSD/resolve/main/data/de/dev_de.jsonl`, `https://huggingface.co/datasets/ZurichNLP/SwissGov-RSD/resolve/main/data/fr/dev_fr.jsonl`, and `https://huggingface.co/datasets/ZurichNLP/SwissGov-RSD/resolve/main/data/it/dev_it.jsonl`, in id order 0 through 223. Label fields are dropped. The ridge is not refit.

```bash
make predict-test
```

On 14 Oct 2026 that command matched the committed test and full files byte for byte. `make run` is the one Docker grade.

## License

Code: Apache-2.0. Report: CC-BY-4.0. SwissGov text stays CC-BY-4.0. See `NOTICE`.
