# Phase 1 validation report

Primary original-post model: UNAVAILABLE. Original-post training outcomes are all zero; the training 90th percentile is zero. Inclusive boundary ties label every record positive. No primary model or performance metrics were fabricated.

Separately trained repost diagnostic, deduplicated by original source ID (not future original-post prediction):
- Model: random_forest_leaf20
- Operational threshold: 0.14
- Historical coverage: 27 seconds
- Test recall: 0.383721; precision: 0.256809; AP: 0.269888
- Test false negatives: 106; false positives: 191

Independent retraining reproduced metrics, target and selection JSON, feature defaults and split manifests. Probabilities matched within absolute tolerance 1e-14.

Streamlit AppTest passed initial state, diagnostic charts, threshold slider, prediction and reset. HTTP health check returned 200.

Local validation uses Python 3.11.0. Python 3.12 was not tested. The installed scikit-learn 1.2.2 / pandas 3.0.0 combination emits deprecation warnings. See pytest-results.txt. Profile snapshot timing is unverified; publication-time validity is not established.

Both post wrapper and original source IDs are deduplicated in the repost diagnostic. Original IDs cannot cross partitions. Hashes remain pseudonymous audit identifiers.
