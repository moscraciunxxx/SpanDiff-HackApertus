# SpanDiff demo script

Spoken length: at most 2 minutes. Visual walkthrough file: `track_2a/docs/demo.html`.

## Say

SpanDiff is the Track 2A entry on SwissGov-RSD. Opening `docs/demo.html`, the scorer is Apertus v1.5 8B at hidden state 15. Each document side is encoded alone in 512-token windows. Following our pre-registered paper-order subword-first formulation, each subword is scored by one minus its maximum cosine similarity to the other side, clamped between zero and one, and averaged inside each whitespace token. No text is generated.

In the results table at the top of `demo.html`, the official development script with `--split dev` printed Spearman de 0,309, fr 0,199, it 0,333, for a four-decimal mean of 0.2803. Our earlier word-mean-pool hidden state 15 ablation in `predictions/l15` printed de 0,296, fr 0,191, it 0,324, mean 0.2703. The clamped final-layer recipe on the same model printed 0,053, 0,056, and 0,106, mean 0.0717. That row is the SwissGov paper baseline on this same model. The paper does not clamp. Hidden state 8 is an ablation at 0,245, 0,165, and 0,282, mean 0.2307.

Scrolling down `demo.html` shows three development pairs side by side — `admin_de_105`, `admin_fr_141`, and `admin_it_101` — comparing SpanDiff token predictions against human gold annotations in English, German, French, and Italian.

The development forward pass used 564,349 tokens and zero generated tokens. Test and full-set predictions are included for the organizers to score; we never graded the test split. Apertus 70B was not run. Judges run make run. That grades the committed files, prints the mean, and does not load the model.
