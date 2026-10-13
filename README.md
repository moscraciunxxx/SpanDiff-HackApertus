# SpanDiff

Repository: https://github.com/moscraciunxxx/SpanDiff-HackApertus

SpanDiff scores token-level semantic drift on Swiss federal pages for Track 2A (UZH). It reads hidden state 15 of `swiss-ai/Apertus-v1.5-8B`, mean-pools subwords into whitespace tokens, and writes `clamp(1 - max cosine, 0, 1)`. No text is generated. Apertus 70B was not run.

## Results

Development split, official script, `--split dev`. The mean is the unweighted average of the three printed Spearmans, to 4 decimals.

| Setup | de | fr | it | mean |
| --- | --- | --- | --- | --- |
| Paper DiffAlign recipe on Apertus 8B, final layer, clamped | 0.053 | 0.056 | 0.106 | 0.0717 |
| SpanDiff ablation, hidden state 8 | 0.245 | 0.165 | 0.282 | 0.2307 |
| SpanDiff, hidden state 15 | 0.296 | 0.191 | 0.324 | 0.2703 |

`track_2a/predictions/l15` is a byte-identical copy of `track_2a/predictions/dev`. Development scores are our development evaluation. Test predictions, 56 documents per language, and full-set predictions, 224 documents per language, use the frozen hidden-state-15 method and are included for the organizers to score. We never graded the test split. `make run` grades `track_2a/predictions/dev` only. Printed development form, decimal commas: de 0,296 ; fr 0,191 ; it 0,324. Kendall de 0,238 ; fr 0,154 ; it 0,259. Source: `track_2a/docs/official_dev_stdout_L15.txt`. The official script does not print that mean. `make run` prints it after the table. The layer-8 and final-layer rows are in `track_2a/docs/official_dev_stdout.txt` and `track_2a/docs/official_dev_stdout_last.txt`.

## Efficiency

Development forward pass: 564,349 tokens, 0 generated. Layer-15 scoring took 515 seconds on an Apple GPU after the weights were cached.

## Run

Judges run this from the repository root. It builds the Docker image and grades `track_2a/predictions/dev` with `--split dev`. It does not load the model.

```bash
make run
```

`make run-local` is that same grade when Docker is not installed. Install the grade dependencies with `pip install -r track_2a/requirements-grade.txt` (the existing pins, plus torch and transformers). `LLM_NAME`, `LLM_BASE_URL`, and `LLM_API_KEY` are not the scorer. An earlier image was built and graded under podman on linux/amd64. Docker is not installed on this Mac, so the current image has not been rebuilt here.

`make predict` loads Apertus and writes a new layer-15 run to `track_2a/predictions/rerun`. It does not replace `track_2a/predictions/dev`. It needs a Hugging Face token only if the weights are not cached, and about 16 GB.

```bash
make predict
```

## License

Code: Apache-2.0. Report: CC-BY-4.0. SwissGov text stays CC-BY-4.0. See `NOTICE`.
