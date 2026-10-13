# SpanDiff

Repository: https://github.com/moscraciunxxx/SpanDiff-HackApertus

Track 2A, UZH. Token-level difference on Swiss federal pages with Apertus v1.5 8B, hidden state 15. Mean-pool subwords to whitespace tokens. Score = clamp(1 - max cosine to the other side, 0, 1). Generated tokens: 0. 70B was not run.

## Results

Development `--split dev`. Mean is the unweighted average of the three printed Spearmans, to 4 decimals.

| Setup | de | fr | it | mean |
| --- | --- | --- | --- | --- |
| Paper DiffAlign recipe on Apertus 8B, final layer, clamped | 0.053 | 0.056 | 0.106 | 0.0717 |
| SpanDiff ablation, hidden state 8 | 0.245 | 0.165 | 0.282 | 0.2307 |
| SpanDiff, hidden state 15 | 0.296 | 0.191 | 0.324 | 0.2703 |

`predictions/l15` is a byte-identical copy of `predictions/dev`. Development scores are our development evaluation. Test predictions, 56 documents per language, and full-set predictions, 224 documents per language, use the frozen hidden-state-15 method and are included for the organizers to score. We never graded the test split. `make run` grades `predictions/dev` only. Development print: de 0,296 ; fr 0,191 ; it 0,324. Files: `predictions/dev/SpanDiff_admin_{de,fr,it}.jsonl.jsonl`. Stdout: `docs/official_dev_stdout_L15.txt`. The official script does not print that mean. `make run` prints it after the table.

## Efficiency

564,349 development forward tokens. 0 generated. 515 seconds of layer-15 scoring on an Apple GPU.

## Run

Judges run this from the repository root:

```bash
make run
```

That builds the Docker image and grades the committed development files. It does not load weights. `make run-local` is that same grade when Docker is not installed. `LLM_NAME`, `LLM_BASE_URL`, and `LLM_API_KEY` are not the scorer. An earlier image was built and graded under podman on linux/amd64. Docker is not installed on this Mac, so the current image has not been rebuilt here.

```bash
make predict
```

Writes a fresh layer-15 development run to `predictions/rerun`. Needs a Hugging Face token only if the weights are not cached, and about 16 GB.

## License

Code Apache-2.0. Report CC-BY-4.0. SwissGov text stays CC-BY-4.0.
