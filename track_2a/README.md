# SpanDiff

Repository: https://github.com/moscraciunxxx/SpanDiff-HackApertus

Track 2A, UZH. Token-level difference on Swiss federal pages with Apertus v1.5 8B, hidden state 15. Mean-pool subwords to whitespace tokens. Score = clamp(1 - max cosine to the other side, 0, 1). Generated tokens: 0. 70B was not run.

## Results

Development `--split dev`. Mean is the unweighted average of the three printed Spearmans, to 3 decimals.

| Setup | de | fr | it | mean |
| --- | --- | --- | --- | --- |
| Paper DiffAlign recipe on Apertus 8B, final layer, clamped | 0.053 | 0.056 | 0.106 | 0.072 |
| SpanDiff ablation, hidden state 8 | 0.245 | 0.165 | 0.282 | 0.231 |
| SpanDiff, hidden state 15 | 0.296 | 0.191 | 0.324 | 0.270 |

Submission print: de 0,296 ; fr 0,191 ; it 0,324. Files: `predictions/dev/SpanDiff_admin_{de,fr,it}.jsonl.jsonl`. Stdout: `docs/official_dev_stdout_L15.txt`.

## Efficiency

564,349 development forward tokens. 0 generated. 515 seconds of layer-15 scoring on an Apple GPU.

## Run

```bash
make run-local
```

Primary path. It grades the committed development files and does not load weights. `make run` is the Docker build. That image was built and graded under podman on linux/amd64. It has not been run on this Mac.

```bash
make predict
```

Writes a fresh layer-15 development run to `predictions/rerun`. Needs a Hugging Face token only if the weights are not cached, and about 16 GB.

## License

Code Apache-2.0. Report CC-BY-4.0. SwissGov text stays CC-BY-4.0.
