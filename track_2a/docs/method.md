# SpanDiff method

Track 2A. SpanDiff is a representation scorer, not a prompt. The submitted model is swiss-ai/Apertus-v1.5-8B only, reading hidden state 15 (index 0 is the embedding). Each side is encoded alone in 512-token windows. Following the pre-registered Tier-1 Combination (docs/preregistration.md), subword vectors at hidden state 15 undergo document-mean contrastive centering (beta = 0.50), bidirectional CSLS hubness margin adjustment (alpha = 0.50, k = 5), and sentence-constrained alignment blending (lambda = 0.35, matched sentence +/- 1), followed by paper-order subword-first clamp(1 - max cosine, 0, 1) pooled into whitespace tokens and window-2 triangular span smoothing over content tokens, rounded to 6 decimals. Generated tokens = 0. Apertus 70B was not run.

Development scores are our development evaluation. Official script, development split: Spearman de 0,373 ; fr 0,231 ; it 0,398. Unweighted mean 0.3340. Kendall de 0,302 ; fr 0,187 ; it 0,320. Each development file has 168 lines. Forward-pass tokens on that run: 564349. Scoring time: 280.2 seconds. Test predictions, 56 documents per language, and full-set predictions, 224 documents per language, use the frozen Apertus hidden-state-15 Tier-1 Combination method and are included for the organizers to score. We never graded the test split. make run grades the development files only.

On the same official command, the unsmoothed subword-first hidden-state-15 ablation printed de 0,309 ; fr 0,199 ; it 0,333 ; mean 0.2803. The word-mean-pool hidden-state-15 ablation (predictions/l15) printed de 0,296 ; fr 0,191 ; it 0,324 ; mean 0.2703. The hidden-state-8 ablation printed de 0,245 ; fr 0,165 ; it 0,282 ; mean 0.2307. As a development comparison only (never submitted), Qwen/Qwen2.5-VL-7B-Instruct with the same Tier-1 Combination printed de 0,351 ; fr 0,217 ; it 0,381 ; mean 0.3163, below Apertus v1.5 8B at 0.3340.

The UZH challenge asks for a SwissGov-paper baseline on this same Apertus model. That baseline is the final-layer DiffAlign row, mean 0.0717 (de 0,053 ; fr 0,056 ; it 0,106). On kept development tokens, no final-layer maximum cosine is negative (maximum kept difference 0.9746), so the unclamped and clamped final-layer paper DiffAlign scores are identical on all 351,942 kept tokens (mean 0.0717). The submitted mean is 0.3340. The UZH FAQ says either the 8B or the 70B is in scope.

Layer 15 and the Tier-1 Combination were chosen via pre-registered selection on dev/train (134/lang, 0.3383 vs 0.2843) and confirmation on dev/val (34/lang, 0.3115 vs 0.2592), beating the +0.0030 full-dev adoption gate by +0.0537 (docs/layer_curve.png, docs/error_analysis.png).

Windows are at most 512 tokenizer tokens. The development pass is 1619 windows. Each window has one leading BOS, and that token is not pooled. EOS, PAD, UNK, and multimodal special counts on that census were 0.

Encoder lines contain only id, text_a, text_b, labels_a, and labels_b. Labels are in [0, 1]. The writer rejects -1 and NaN. Label length equals the whitespace-token count. LLM_NAME, LLM_BASE_URL, and LLM_API_KEY are not read. The scores come from a local forward pass of Apertus v1.5 8B.

Code is Apache-2.0. The report is CC-BY-4.0. SwissGov text stays CC-BY-4.0.
