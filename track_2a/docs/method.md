# SpanDiff method

Track 2A. SpanDiff is a representation scorer, not a prompt. The model is swiss-ai/Apertus-v1.5-8B. The submitted state is hidden state 15 (index 0 is the embedding). Each side is encoded alone. Following the pre-registered paper-order subword-first formulation (docs/preregistration.md), each content subword is scored by clamp(1 - max cosine to the other side, 0, 1), and subword scores are averaged inside the whitespace token they overlap most, rounded to 6 decimals. Generated tokens = 0. Apertus 70B was not run.

Development scores are our development evaluation. Official script, development split: Spearman de 0,309 ; fr 0,199 ; it 0,333. Unweighted mean 0.2803. Kendall de 0,249 ; fr 0,161 ; it 0,266. Each development file has 168 lines. Forward-pass tokens on that run: 564349. Scoring time: 549.8 seconds. Test predictions, 56 documents per language, and full-set predictions, 224 documents per language, use the frozen hidden-state-15 subword-first method and are included for the organizers to score. We never graded the test split. make run grades the development files only.

On the same official command, the word-mean-pool hidden-state-15 ablation (predictions/l15) printed de 0,296 ; fr 0,191 ; it 0,324 ; mean 0.2703. Pre-registered variant (a), averaging hidden states 12-17 with word mean-pooling, printed de 0,285 ; fr 0,183 ; it 0,316 ; mean 0.2613. Pre-registered variant (c), combining hidden states 12-17 with subword-first scoring, printed de 0,298 ; fr 0,192 ; it 0,325 ; mean 0.2717. The hidden-state-8 ablation printed de 0,245 ; fr 0,165 ; it 0,282 ; mean 0.2307. The final-layer clamped DiffAlign recipe printed de 0,053 ; fr 0,056 ; it 0,106 ; mean 0.0717. Those ablation files are not the primary submission.

The UZH challenge asks for a SwissGov-paper baseline on this same Apertus model. That baseline is the clamped final-layer row, mean 0.0717. The submitted mean is 0.2803 (and 0.2703 with word-first mean-pooling). The paper does not clamp. An unclamped rescore was not produced. The UZH FAQ says either the 8B or the 70B is in scope.

Layer 15 was chosen because the saved per-language development map peaks in the middle of the network, and pre-registered variant (b) beat word-mean-pool L15 on dev/train (0.2843 vs 0.2734), dev/val (0.2592 vs 0.2527), and the full-development official grade (+0.0100, exceeding the +0.0050 threshold). The map is in docs/layer_curve.png.

Windows are at most 512 tokenizer tokens. The development pass is 1619 windows. Each window has one leading BOS, and that token is not pooled. EOS, PAD, UNK, and multimodal special counts on that census were 0.

Encoder lines contain only id, text_a, text_b, labels_a, and labels_b. Labels are in [0, 1]. The writer rejects -1 and NaN. Label length equals the whitespace-token count. LLM_NAME, LLM_BASE_URL, and LLM_API_KEY are not read. The scores come from a local forward pass of Apertus v1.5 8B.

Code is Apache-2.0. The report is CC-BY-4.0. SwissGov text stays CC-BY-4.0.
