"""SpanDiff: Apertus v1.5 8B token-level semantic difference on SwissGov-RSD dev."""

MODEL_ID = "swiss-ai/Apertus-v1.5-8B"
TRANSFORMERS_COMMIT = "3797303dda74844e3d1f8977ff5518bb91f818b4"
PREDICTION_PREFIX = "track_2a/predictions/dev/SpanDiff_admin_"
GRADED_COMMAND = (
    "python -m scripts.evaluate_predictions_admin "
    "track_2a/predictions/dev/SpanDiff_admin_ "
    "--split dev"
)
