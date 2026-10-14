# SpanDiff demo script

Spoken length: at most 2 minutes. Visual walkthrough file: `track_2a/docs/demo.html`.

## Say

SpanDiff is the Track 2A entry on SwissGov-RSD. Opening `docs/demo.html`, the submitted scorer is Apertus v1.5 8B at hidden state 15. Each document side is encoded alone in 512-token windows. Following our pre-registered Tier-1 Combination, subword states undergo document-mean contrastive centering, bidirectional CSLS hubness margin adjustment, sentence-constrained alignment blending, paper-order subword-first clamp of one minus maximum cosine, and window-2 triangular span smoothing over content tokens. No text is generated.

In the results table at the top of `demo.html`, the official development script with `--split dev` printed Spearman de 0,373, fr 0,231, it 0,398, for a four-decimal mean of 0.3340. Unsmoothed subword-first hidden state 15 printed de 0,309, fr 0,199, it 0,333, mean 0.2803. Word-mean-pool hidden state 15 in `predictions/l15` printed de 0,296, fr 0,191, it 0,324, mean 0.2703. The final-layer recipe on the same model printed 0,053, 0,056, and 0,106, mean 0.0717, both clamped and unclamped. That row is the SwissGov paper baseline on this same model. Hidden state 8 is an ablation at 0,245, 0,165, and 0,282, mean 0.2307, and a local Qwen2.5-VL-7B comparison row with the same method scored 0.3163.

Scrolling down `demo.html` shows three development pairs side by side — `admin_de_105`, `admin_fr_141`, and `admin_it_101` — comparing SpanDiff token predictions against human gold annotations in English, German, French, and Italian.

The development forward pass used 564,349 tokens and zero generated tokens. Test and full-set predictions are included for the organizers to score; we never graded the test split. Apertus 70B was not run. Judges run make run. That grades the committed files, prints the mean, and does not load the model.
