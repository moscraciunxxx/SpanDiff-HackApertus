# SpanDiff method

Track 2A. SpanDiff is a representation scorer, not a prompt. The model is swiss-ai/Apertus-v1.5-8B. The submitted state is hidden state 15 (index 0 is the embedding). Each side is encoded alone. Subwords are mean-pooled into the whitespace token they overlap most. The score is clamp(1 - max cosine to the other side, 0, 1), written to 6 decimals. Generated tokens = 0. Apertus 70B was not run.

Development result from the official script with --split dev: Spearman de 0,296 ; fr 0,191 ; it 0,324. Unweighted mean 0.270. Kendall de 0,238 ; fr 0,154 ; it 0,259. Each file has 168 lines. Forward-pass tokens on that run: 564349. Scoring time: 515 seconds.

The same command on the hidden-state-8 ablation printed de 0,245 ; fr 0,165 ; it 0,282 ; mean 0.231. The final-layer clamped DiffAlign recipe printed de 0,053 ; fr 0,056 ; it 0,106 ; mean 0.072. Those files are not the submission.

Layer 15 was chosen because the saved per-language development map peaks in the middle of the network, and the official development grade is higher at 15 than at 8. The map is in docs/layer_curve.png. It is an average-rank study, not a second official score.

Windows are at most 512 tokenizer tokens. The development pass is 1619 windows. Each window has one leading BOS, and that token is not pooled. EOS, PAD, UNK, and multimodal special counts on that census were 0.

Encoder lines contain only id, text_a, text_b, labels_a, and labels_b. Labels are in [0, 1]. The writer rejects -1 and NaN. Label length equals the whitespace-token count.

Code is Apache-2.0. The report is CC-BY-4.0. SwissGov text stays CC-BY-4.0.
