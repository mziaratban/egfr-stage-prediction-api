from prometheus_client import Counter, Histogram, Gauge


# ==================================
# Prediction Failures
# ==================================

from prometheus_client import Counter


prediction_total = Counter(
    "prediction_total",
    "Total number of predictions"
)


prediction_stage_counter = Counter(
    "prediction_stage_total",
    "Number of predictions by eGFR stage",
    ["day", "stage"]
)


prediction_failures = Counter(
    "prediction_failures_total",
    "Number of failed predictions"
)

# ==================================
# Active Predictions
# ==================================

active_predictions = Gauge(
    "prediction_in_progress",
    "Number of predictions currently running"
)


# ==================================
# Prediction Latency
# ==================================

prediction_duration = Histogram(
    "prediction_duration_seconds",
    "Prediction inference latency in seconds",
    buckets=(
        0.001,
        0.005,
        0.01,
        0.02,
        0.05,
        0.1,
        0.2,
        0.5,
        1,
        2,
    ),
)