# SpanDiff

SpanDiff scores token-level semantic drift on Swiss federal pages for Track 2A (UZH). It reads hidden state 15 of `swiss-ai/Apertus-v1.5-8B`, mean-pools subwords into whitespace tokens, and writes `clamp(1 - max cosine, 0, 1)`. No text is generated. Apertus 70B was not run.

## Results

Development split, official script, `--split dev`. The mean is the unweighted average of the three printed Spearmans, to 3 decimals.

| Setup | de | fr | it | mean |
| --- | --- | --- | --- | --- |
| Paper DiffAlign recipe on Apertus 8B, final layer, clamped | 0.053 | 0.056 | 0.106 | 0.072 |
| SpanDiff ablation, hidden state 8 | 0.245 | 0.165 | 0.282 | 0.231 |
| SpanDiff, hidden state 15 | 0.296 | 0.191 | 0.324 | 0.270 |

Printed form of the submission, decimal commas: de 0,296 ; fr 0,191 ; it 0,324. Kendall de 0,238 ; fr 0,154 ; it 0,259. Source: `track_2a/docs/official_dev_stdout_L15.txt`. The layer-8 and final-layer rows are in `official_dev_stdout.txt` and `official_dev_stdout_last.txt`.

## Efficiency

Development forward pass: 564,349 tokens, 0 generated. Layer-15 scoring took 515 seconds on an Apple GPU after the weights were cached.

## Run

From the repository root:

```bash
make run-local
```

That grades `track_2a/predictions/dev` with `--split dev` and does not load the model. Install the grade dependencies with `pip install -r track_2a/requirements-grade.txt` (the existing pins, plus torch and transformers). `make run` builds the Docker image when Docker is installed. The image was built and graded under podman on linux/amd64. It has not been run on this Mac.

Regenerate development predictions (Hugging Face token if the weights are not cached, about 16 GB):

```bash
make predict
```

## License

Code: Apache-2.0. Report: CC-BY-4.0. SwissGov text stays CC-BY-4.0. See `NOTICE`.
