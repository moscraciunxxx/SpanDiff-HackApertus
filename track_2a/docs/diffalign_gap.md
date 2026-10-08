# DiffAlign in the ACL 2026 paper, and SpanDiff

Source: section 4.1 and section 5.2 of https://aclanthology.org/2026.acl-long.1437.pdf. Those sections produce the unsupervised Table 3 rows, including LaBSE, XLM-R+SimCSE, and Qwen3-Embedding. The numbers in that table are the paper's, on the full SwissGov set, and the table prints Spearman times 100. They are not a SpanDiff score. SpanDiff's development scores are not on that scale.

## What the paper does

1. Encode text A and text B separately.
2. Take the final hidden state of each encoded token.
3. For token a_i, the score is 1 minus the maximum cosine between that state and any token state in B. The same step runs from B back to A.
4. The paper writes no clamp. A negative cosine therefore gives a score above 1.
5. No labeled training is used for these rows.
6. The encoders named for the rows are XLM-R+SimCSE and LaBSE, following Vamvas and Sennrich (2023), plus longer encoders that include Qwen3-Embedding. XLM-R and LaBSE are limited to 512 tokens. Qwen3-Embedding is described as taking a whole document without splitting, because its context is 8,192 tokens. This paper does not describe how a 512-token encoder is split, and it does not describe mean-pooling subwords onto whitespace tokens.

## SpanDiff

SpanDiff keeps the same maximum: each vector is scored against every vector on the other side, and the best cosine is the one that counts. It then applies clamp(1 - that cosine, 0, 1). Before that, it mean-pools Apertus v1.5 8B hidden state 15 into whitespace tokens. Index 0 is the embedding. The submitted score stays on those Apertus states. Hidden state 8 is the ablation. LaBSE and the other Table 3 encoders are not a second scorer.

The methods differ in order as well as in the clamp. The paper takes the maximum cosine at subword level, then averages those scores inside each word. SpanDiff mean-pools the hidden states into the word first, then takes the maximum cosine. On the full development pass, clamped subword-then-mean scored 0.2621, 0.1750, and 0.2928 against clamped mean-pool 0.2495, 0.1662, and 0.2842. The graded files keep the pooled order.

Dropping the clamp, on the same pooled Apertus states, changes a token only when its best cosine is negative: the paper formula returns a number above 1, and SpanDiff returns 1.

The graded development print is hidden state 15 after mean-pooling, with the clamp: Spearman de 0,296 ; fr 0,191 ; it 0,324 ; unweighted mean 0.270. Hidden state 8 printed 0,245 ; 0,165 ; 0,282 ; mean 0.231. The higher pooling numbers above are ablations. They were not written into the graded files.
