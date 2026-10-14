# SpanDiff

Repository: https://github.com/moscraciunxxx/SpanDiff-HackApertus

Track 2A, UZH. Token-level difference on Swiss federal pages with Apertus v1.5 8B, hidden state 15. Paper-order subword-first scoring: `clamp(1 - max cosine to the other side, 0, 1)` per subword, then mean-pooled inside each whitespace token. Generated tokens: 0. 70B was not run.

## Results

Development `--split dev`. Mean is the unweighted average of the three printed Spearmans, to 4 decimals.

| Setup (all on `swiss-ai/Apertus-v1.5-8B`) | de | fr | it | mean |
| --- | --- | --- | --- | --- |
| Paper DiffAlign recipe on Apertus 8B, final layer, clamped | 0.053 | 0.056 | 0.106 | 0.0717 |
| SpanDiff ablation, hidden state 8 (word mean-pool) | 0.245 | 0.165 | 0.282 | 0.2307 |
| Pre-registered (a): mean of hidden states 12–17 (word mean-pool) | 0.285 | 0.183 | 0.316 | 0.2613 |
| SpanDiff ablation, hidden state 15 (`predictions/l15`, word mean-pool) | 0.296 | 0.191 | 0.324 | 0.2703 |
| Pre-registered (c): hidden states 12–17 + subword-first | 0.298 | 0.192 | 0.325 | 0.2717 |
| **SpanDiff submitted, hidden state 15 (paper-order subword-first)** | **0.309** | **0.199** | **0.333** | **0.2803** |

Development scores are our development evaluation. Test predictions, 56 documents per language, and full-set predictions, 224 documents per language, use the frozen hidden-state-15 subword-first method and are included for the organizers to score. We never graded the test split. `make run` grades `predictions/dev` only. `predictions/l15` retains the word-mean-pool hidden-state-15 ablation (`0.2703`). Development print: de 0,309 ; fr 0,199 ; it 0,333 (Kendall de 0,249 ; fr 0,161 ; it 0,266). Files: `predictions/dev/SpanDiff_admin_{de,fr,it}.jsonl.jsonl`. Stdout: `docs/official_dev_stdout_subword_l15.txt`. The official script does not print that mean. `make run` prints it after the table. Static visual demo: `docs/demo.html`.

## Efficiency

Development forward pass: 564,349 tokens, 0 generated, 549.8 seconds on an Apple GPU with cached weights.

## Run

Judges run this from the repository root:

```bash
make run
```

That builds the Docker image and grades the committed development files. It does not load weights. `make run-local` is that same grade when Docker is not installed. `LLM_NAME`, `LLM_BASE_URL`, and `LLM_API_KEY` are not the scorer. An earlier image was built and graded under podman on linux/amd64. Docker is not installed on this Mac, so the current image has not been rebuilt here. Windows Docker Desktop verification commands are in `docs/DOCKER_CHECK.md`.

```bash
make predict
```

Writes a new deterministic layer-15 run to `predictions/rerun`. Does not replace `predictions/dev`.

## License

Code: Apache-2.0. Report: CC-BY-4.0. SwissGov text stays CC-BY-4.0.
