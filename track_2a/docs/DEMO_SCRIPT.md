# SpanDiff demo script

Spoken length: at most 2 minutes.

## Say

SpanDiff is the Track 2A entry. The scorer is Apertus v1.5 8B, hidden state 15. Each side is encoded alone. Subwords are mean-pooled into whitespace tokens. The score is one minus the best cosine, clamped between zero and one. No text is generated.

On the development split the official script printed Spearman de 0,296, fr 0,191, it 0,324. The mean is 0.2703. The clamped final-layer recipe on the same model printed 0,053, 0,056, and 0,106. That row is the SwissGov paper baseline on this same model. The paper does not clamp. Hidden state 8 is an ablation at 0,245, 0,165, and 0,282.

The forward pass used 564,349 tokens. Generated tokens were zero. Apertus 70B was not run. Judges run make run. That grades the committed files, prints the mean, and does not load the model.
