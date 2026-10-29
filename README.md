# Tomato AI

## Launch the integrated Command Center

```powershell
streamlit run app.py
```

The existing root application now has six sections:

| Section | Contents |
| --- | --- |
| Command Center | Actual active-session counts, recent activity, shared Analyze Post workflow |
| ML Spread Lab | Frozen final-model charts, threshold explorer, metadata inference; other tested models and legacy Combined Analysis in separate subviews |
| LLM Evaluation Lab | Existing GPT-OSS-120B benchmarks, provisional versus verified references, confusion matrices, annotation and classification Try It Yourself |
| RAG & Response Lab | Actual policy-index status, saved retrieval/response evaluation, citations and shared Analyze Post |
| Crisis Monitoring & Reviewer Queue | Prioritized decisions, rule explanations, explicit approve/edit/reject/escalate controls and persistent history |
| Demo & Audit | Reproducible seven-day feed, four scenarios, processing/version audit, safe session reset |

The feed references 750 original records from the unchanged 15,000-post dataset plus 3 reused separate language/safety examples. It does not run inference automatically. English language hints reflect source templates; Hindi/Hinglish references remain provisional. Display dates map onto January 1–7 2026 UTC, with original dates retained and excluded from presentation-time ML changes. Demo selection includes source training records and is not an independent benchmark; model metrics continue to come from saved held-out evaluations.

Analyze one selected or custom post explicitly. Blank metadata remains missing, so text-only submissions have no invented ML score. Real-output cache keys bind models, prompts, ML manifest, policy ingestion/chunks, endpoint fingerprint, embedding revision and decision implementation. Results identify cached output; live rerun bypasses pipeline and classifier caches. An external Qdrant mutation outside the ingestion workflow requires explicit rerun or an updated ingestion manifest.

Demo processing and human review use `data/demo_state/<epoch>/review.sqlite3`; existing global history remains in `data/review/decisions.sqlite3`. Reset starts a new epoch and keeps all previous records accessible through the reviewer source selector. No source data, verified annotations, models, evaluations or policy collections are deleted. Approval never publishes. Reviewer identity remains locally asserted rather than authenticated.

Credentials are read from Streamlit secrets with environment fallback. Local credential stores are excluded from Git. Do not put keys in code, exports or reports. No new dependencies were required for Phases 5/6; install the existing `requirements.txt` if preparing a new environment. This full stack was tested on this workspace's Python 3.11; Python 3.12 full-stack verification remains pending.

See [Phase 5 report](phase56_results/PHASE5_REPORT.md), [final integration report](phase56_results/FINAL_REPORT.md), and [test results](phase56_results/tests_all.xml). Run `python -m pytest -q` for regression tests. Historical offline retraining commands below are optional reproduction instructions, not required to launch the application. Phases 5/6 did not retrain or regenerate data. Phase 7 has not been started.

## Phase 1.5 final selected ML model

The ML dashboard displays only the final selected model. The active pointer is
`artifacts/ml_active.json`; it selects one immutable directory under
`artifacts/ml_versions/`. A hash manifest binds that version's model, preprocessing,
feature order, decision threshold, metrics and all chart data. Old ML artifacts and
LLM functionality are preserved. The historical dataset report has its own ML tab.

```powershell
python -m ml.retrain --iterations 8
python -m ml.diagnostics
python -m pytest tests -q
streamlit run app.py
```

Retraining uses the existing 15,000-post simulation-v1 data and existing chronological
9,000/3,000/3,000 split. It does not regenerate data or change the target. Five model
families are tuned with training-only author-purged temporal CV; final selection
uses validation average precision. The operational threshold now maximizes
validation F1 on 0.01–0.99, with higher thresholds breaking exact ties. This replaces
the earlier cost-based operational objective. The test slider is descriptive only;
it recalculates precision, recall, F1/F2 and the confusion matrix from stored test
predictions and never changes the deployment threshold.

The test sets were already examined in earlier phases. This run keeps them out of
model/threshold selection and evaluates the frozen winner once; it cannot honestly
claim a newly untouched final test. The current model's stronger precision and
lower recall are a threshold trade-off, not an improvement in ranking performance.
The simulator's latent appeal, platform amplification and late shocks create
substantial overlap. Training-only plots and sample-size diagnostics are saved in
`phase15_results/diagnostic_plots.html`; no Bayes performance ceiling is claimed.

Each version contains the full evaluation report, audit, split manifest, feature
schema, CV results, final predictions, precision–recall/ROC/calibration CSVs,
confusion counts, feature importance and evaluation timestamp. The final human
summary is `phase15_results/FINAL_REPORT.md`. Candidate comparisons are available under ML Spread Lab → Other tested models, separately from the final-model results.

The Plotly figures use ordinary numeric lists rather than Plotly 6 binary array
payloads, which older Streamlit browser bundles could not render correctly.
All six final-model charts are visible and labelled; missing or corrupt artifacts
produce explicit errors. Python 3.12 primary-model/chart checks are recorded in
`phase15_results/python312-tests.txt`.

For deployment, provision `artifacts/ml_active.json`, its referenced complete bundle,
and `artifacts/ml/data_report.json` together. Model files are Git-ignored; do not
publish `.env`, secrets or customer data. The app loads saved files and does not
train on page load. Run training offline in the recorded dependency environment.
The existing legacy `ml.train`/`ml.tune` commands reproduce old experiments and do
not replace the new active pointer; use `ml.retrain` for current deployment.

## Integrated application

From this project directory:

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

Use the six-section sidebar described above. Classification Try It Yourself is in **LLM Evaluation Lab**; legacy **Combined Analysis** remains under **ML Spread Lab**. The existing spread model and all ML
data/artifacts remain unchanged; integration does not retrain anything. The LLM
package reuses the existing AIAgent prompt, taxonomy and Groq model
`openai/gpt-oss-120b`, with `GROQ_MODEL` available as a configuration override.

The project root `.env` supplies `GROQ_API_KEY` and `GROQ_MODEL` for the LLM.
Existing process variables are preserved; Streamlit secrets remain supported.
Live submissions are isolated from the imported benchmark and are saved only
when explicitly selected. Combined Analysis requires real user-supplied metadata;
blank fields produce no ML probability. Severe LLM flags require human review
regardless of spread. No RAG or response generation is included.

```text
app.py                         Existing ML dashboard plus sidebar navigation
config.py                      Shared secrets/env resolver
llm/
  classifier.py                Adapted existing Groq classifier
  prompts/classifier_prompt.txt  Original prompt, unchanged
  evaluation_core.py           Reused original evaluation functions
  evaluate.py                  Isolated offline/live benchmark runner
  combined.py                  Independent inference and review routing
  ui.py                        LLM, evaluation and combined sections
ml/                            Existing spread implementation, unchanged
artifacts/                     Existing ML models/reports, unchanged
data/llm/
  benchmark/                   Evaluated reference/prediction records and reports
  source_snapshot/             Original mixed files retained for provenance
  imported_live/               Legacy live rows, never reference ground truth
  live/                        Opt-in new submissions and local backups
  evaluations/                 Separate new evaluation runs
AIAgent/                       Original project, preserved unchanged
tests/                         Existing ML tests and new integration tests
```

Run `python -m pytest tests -q` and `python -m llm.evaluate` to verify the
integration without new API calls. To classify a post outside Streamlit:
`python -m llm.classifier --post "My fictional order arrived cold."`.
For the tested Windows Python 3.12 package versions, use
`py -3.12 -m pip install -r requirements.txt -c constraints-python312.txt`.
Detailed migration choices, Python 3.12 compatibility evidence and the archived
Random Forest compatibility failure are documented in
[docs/LLM_INTEGRATION.md](docs/LLM_INTEGRATION.md).

## Credentials

In Streamlit, put `GROQ_API_KEY`, `QDRANT_URL`, and `QDRANT_API_KEY` at the
top level of `.streamlit/secrets.toml`, or configure them in Streamlit Cloud's
secrets settings. See `.streamlit/secrets.toml.example` for placeholders.
The real secrets file is excluded by `.gitignore`.

Standalone Python scripts read these same names from the process environment.
`.env.example` documents the names. LLM entry points load the root `.env` without
overwriting process variables; the original credential loader remains available.
In Streamlit, each missing or blank secret falls back to its environment variable.

Use the shared loader wherever clients are constructed:

```python
from config import load_credentials
from groq import Groq
from qdrant_client import QdrantClient

credentials = load_credentials()
groq_client = Groq(api_key=credentials.groq_api_key)
qdrant_client = QdrantClient(
    url=credentials.qdrant_url,
    api_key=credentials.qdrant_api_key,
)
```

Do not print credentials or include them in logs. No external API calls are
made by the loader. The ML dashboard does not require external API credentials.

## ML Spread Prediction · synthetic primary model

The dashboard and shared `ml.inference.SpreadPredictor` load only the synthetic
model in `artifacts/synthetic_ml`. Historical models are never fallback models.
The retained historical original-post limitations report is available at
`artifacts/ml/data_report.json` and described in the evaluation documentation.

From this project directory:

```powershell
python -m pip install -r requirements.txt
python -m ml.train
python -m ml.tune --iterations 8
python -m pytest tests -q
python -m streamlit run app.py
```

`python -m ml.train` generates 15,000 fictional food-delivery posts plus a separate
3,000-post shifted synthetic test set, trains seven candidate configurations
(including a dummy baseline), freezes the validation-selected model/threshold,
and evaluates both tests. It requires no credentials or external API calls.
The generator parameters are fixed, versioned and not adjusted to hit any score.
Full equations, distributions, exclusions and limitations are documented in
[docs/SYNTHETIC_SIMULATION.md](docs/SYNTHETIC_SIMULATION.md).

Prediction uses author metadata, numeric content metadata and **observed first-hour
engagement** to classify eventual simulated high spread. It is available only
after the first hour. Raw text, sentiment/topic/urgency labels, final retweets,
identifiers and hidden simulation variables never enter the model. Text and
reference labels remain in JSONL for LLM evaluation and integrated demonstrations.

### Data and output files

- `data/synthetic/food_delivery.jsonl`: 15,000 fictional posts with text, metadata,
  first-hour engagement, simulated final retweets and reference sentiment/topic.
- `data/synthetic/shifted_test.jsonl`: 3,000 independent shifted-simulation posts.
- `data/synthetic/simulation_manifest.json`: seeds, parameters, whitelist, limits.
- `data/synthetic/*_latent_audit.csv`: isolated hidden simulation draws for audit;
  never used as model inputs.
- `artifacts/synthetic_ml/model.joblib`: fitted preprocessing and classifier.
- `feature_schema.json`, `label_definition.json`, `selection.json`: feature inputs,
  training medians, empirical training-only target cutoff and frozen thresholds.
- `split_manifest.csv`, `data_report.json`: split membership, date coverage,
  prevalence, characteristics and excluded fields.
- `held_out_predictions.csv`, `metrics.json`: actual validation, test and shifted
  predictions/results at default, F1, F2 and operational thresholds.
- `model_comparison_validation.csv`, `thresholds_validation.csv`: model selection
  evidence and threshold tradeoffs. PR-AUC is summarized by average precision.
- `feature_importance.csv`, `calibration.csv`: validation permutation importance
  and equal-width reliability bins for validation and both tests.
- `run.json`, `requirements-training.txt`, `status.json`: reproducibility and status.

Unqualified artifact filenames above live under `artifacts/synthetic_ml`.
Datasets and generated artifacts are ignored by Git. To deploy, generate the
artifacts during a controlled build or provision them privately. Only load trusted
local joblib files: deserialization can execute code. Retrain after dependency
changes rather than loading incompatible scikit-learn artifacts.

To call the same primary model from integrated application code:

```python
from ml.inference import SpreadPredictor
predictor = SpreadPredictor()
result = predictor.predict({
    "followers_count": 450,
    "first_hour_impressions": 900,
    "first_hour_likes": 30,
    "first_hour_replies": 8,
    "first_hour_retweets": 6,
})
```

Partial inputs use training medians. The dashboard exposes every input and
reset controls, local median-replacement sensitivity, threshold exploration,
calibration, final confusion matrices, and shifted-test comparison.

### Reproducibility and historical audit

Python 3.12 is the deployment target. Local execution uses Python 3.11; exact
versions and seeds are recorded in `run.json`. Repeat training to a separate
output directory with the same fixed seeds:

```powershell
python -m ml.train --output artifacts/synthetic_repeat
```

Historical audit commands remain available explicitly, without changing primary
inference. Original-post training has degenerate zero-retweet outcomes and reports
unavailable rather than inventing a model:

```powershell
python -m ml.train --source historical --output artifacts/ml
python -m ml.train --source historical --cohort reposts --output artifacts/repost_diagnostic
```

The repost diagnostic deduplicates nested source-post IDs and is not a substitute
for original-post validation. Both historical metadata provenance and representativeness
remain limited. Synthetic scores are results against a fictional mechanism;
they do not establish real-world predictive validity, reach estimates, or crisis
severity. All customer responses still require human review.

## Tuned ML lab

`python -m ml.tune --iterations 8` retrains the improved model using the existing
fixed synthetic datasets. Run `python -m ml.train` only to generate/rebuild the
baseline first. The tuning command archives the previous artifacts once in
`artifacts/synthetic_previous`, verifies that the original data hashes, target
cutoff and split membership are unchanged, and publishes the chosen pipeline in
`artifacts/synthetic_ml`. Never use the reported test scores to rerun searches
until a preferred score appears.

RandomizedSearchCV compares Logistic Regression, Random Forest, histogram gradient
boosting, XGBoost, LightGBM and CatBoost. Eight sampled configurations per family
use three expanding chronological CV folds inside the training partition. Each
fold removes overlapping authors from validation and keeps timestamp boundaries
separate. All simulation authors are unique; this grouping safeguard does not
prove repeated-author robustness. The full test and validation partitions are
excluded from hyperparameter fitting. Median imputation, log transforms, numerical
feature engineering and scaling are fitted inside each CV pipeline.

Engineered inputs include smoothed likes/replies/retweets per impression,
engagement shares, impressions and retweets per follower, list membership per
follower, and cyclic UTC hour/weekday. All derive only from allowed inputs
available after the first hour. Search spaces include class/positive weights;
there is no oversampling. The simulator and target labels are not modified.

Family winners are ranked by validation average precision. The previous frozen
model remains eligible, so more complicated models are not automatically chosen.
Operational thresholds are selected solely on validation predictions: minimize
`5*FN+FP` with recall at least 0.50 and alert rate at most 0.30 where feasible.
If infeasible, select the highest recall within the alert cap. The old model
retains its previous operational threshold for a fair before/after comparison.
F1- and F2-optimal thresholds and the default 0.50 threshold are also saved.
No model or threshold is selected using test accuracy.

The original test set was inspected during earlier development. Its before/after
results are labelled **Original test set** and must not be called a completely
fresh confirmation. After freezing all candidate models and probability thresholds,
a separate 3,000-post **New confirmation set** is generated with fixed seed
20261031 under the unchanged simulator. Each frozen candidate is evaluated on it
once; the deployed winner is not reselected using those results. Repeated runs
with this same seed are reproducibility checks, not additional fresh tests.
The shifted simulation remains a separate stress test. All results remain
synthetic and do not establish real-world performance.

The dashboard now has three focused tabs: Model metrics, Threshold explorer,
and Try it yourself. It reports accuracy, precision, recall, F1, average precision
(the PR-AUC summary), ROC-AUC, calibration, confusion matrices, feature importance,
error slices and all models' held-out comparisons. The majority classifier is
shown explicitly: high accuracy with zero viral recall is not a useful viral detector.
Threshold exploration uses validation data and does not silently change deployment.

Methodological material and limitations are kept here and in
`artifacts/synthetic_ml/EVALUATION_REPORT.md`, rather than standalone warning panels
in the dashboard. The historical limitations report is preserved unchanged in
`artifacts/ml/data_report.json`; the original synthetic specification remains in
`docs/SYNTHETIC_SIMULATION.md`. The primary integration uses only the saved synthetic
model through `SpreadPredictor` and never the historical repost diagnostic.

Additional outputs under `artifacts/synthetic_ml`:

- `cv_*.csv`: sampled hyperparameters and train/CV scores.
- `tuning.json`: compatibility, best hyperparameters, group/date-fold audit and
  train-minus-CV AP gaps.
- `model_comparison_test.csv`: frozen current, improved and majority models on
  original, shifted and new confirmation sets.
- `error_analysis.csv`: false positives/negatives by author reach and early activity.
- `overfitting.json`: selected pipeline's training versus validation metrics.
- `EVALUATION_REPORT.md`: actual results, the accuracy investigation, test reuse
  disclosure, data leakage audit and limitations.

The independent confirmation records are saved at
`data/synthetic/confirmation_test.jsonl`. Local validation uses Python 3.11;
Python 3.12 compatibility has not been verified in this environment.

## Model comparison page and LLM evaluation charts

The sidebar **ML Model Comparisons** page shows existing saved experiments without
retraining or changing the deployed model. The latest retraining table uses the
validation split and the default 0.50 threshold for every candidate, matching its
saved comparison file. Earlier tuning results are shown separately with original,
confirmation and shifted held-out splits and each model's saved operational threshold.
The page includes accuracy, precision, recall, F1/F2, average precision, ROC-AUC,
balanced accuracy, Brier, confusion counts, selectable metric plots and per-model
confusion matrices. Training-only CV scores and parameters are available for the
latest experiment. Candidate-model probability curves were not saved for every
run; the comparison page does not invent them.

The LLM Evaluation page now uses ordinary-list Plotly traces for both the field
metrics bar chart and all eight selectable confusion matrices. Counts are derived
from the existing reference/prediction records, and each matrix shows its benchmark
sample count. This avoids the Plotly 6 binary-array incompatibility with older
Streamlit frontends. No benchmark results, model weights or decision thresholds
were modified.


## Phase 2 canonical LLM integration (GPT-OSS)

Launch the one application with `streamlit run app.py`. The sidebar retains both
ML pages, LLM Classification, LLM Evaluation and Combined Analysis. The original
`AIAgent/` implementation and historical evaluations remain read-only references.
No ML dataset, split, model weight, operational threshold or ML source is changed.

The reusable entry point is `llm.classifier.classify_post(text, post_id=None)`.
It returns a Pydantic `ClassificationResult` containing either a validated
`classification` or an explicit error, plus model/prompt/input hashes, timestamp,
latency, actual token usage when available, attempts and cache status. The original
agent's official Groq SDK / JSON-object approach and food-delivery taxonomy are
retained. Parsing, schema, persistence and evaluation were refactored because the
old persistence could copy predictions into reference labels. No response generation,
RAG, final publishing decision or new escalation engine is implemented.

`GROQ_MODEL=openai/gpt-oss-120b` is the primary default and preserves the existing
project setting. No switch to Llama is made. Credentials and model configuration
use project-level Streamlit secrets inside Streamlit, then environment variables;
standalone scripts load only the root `.env` without overriding process variables.
The old `AIAgent/.env` is never loaded. Optional `GROQ_CLASSIFIER_TIMEOUT` (default
45 seconds, capped 120) and `GROQ_CLASSIFIER_RETRIES` (default 2, capped 3) control
bounded retries. Rate-limit/server/connection errors back off, respecting numeric
Retry-After up to 30 seconds. Failures never become fabricated classifications.

Canonical fields: sentiment (positive/neutral/negative/mixed), primary topic,
secondary_topics, urgency (low/medium/high/critical), suggested_severity (P0–P3,
advisory), language (English/Hindi/Hinglish/other), sarcasm_detected, safety_flags
(multi-label), response_recommended, evidence_phrases (exact text spans), brief_reason,
and ambiguous. Enums and strict booleans reject unsupported outputs. There is no
required confidence field. Optional model self-reported confidence is stored
separately; it is never shown as a calibrated probability. The versioned prompt is
`prompts/classifier_versions/classifier_v2.0.0.md`; `prompts/classifier.md` is its
readable current copy. The original agent prompt is preserved separately.

Legacy mapping is explicitly lossy: needs_reply becomes response_recommended,
sarcasm becomes sarcasm_detected, threat_legal becomes legal, and a supported
single risk becomes a singleton safety list. Legacy human_review is advisory,
never approval. Legacy critical urgency does not by itself prove imminent P0
harm: mapped severity remains provisional P1. Missing simultaneous risks cannot
be reconstructed. The original six topics remain, with privacy_security and
rider_conduct added. No legacy or generated label is automatically human-verified.

Data layout:

```
llm/                       classifier, schema, config, storage, annotation, evaluation, UI
prompts/                   canonical prompt and immutable versions
AIAgent/                   preserved original source and historical artifacts
data/synthetic/            unchanged 15,000-post ML dataset and other existing splits
data/llm/benchmark/         isolated 50-post historical GPT-OSS benchmark
data/llm_evaluation/        balanced candidates, supplements, human annotations, run reports
data/llm_predictions/       inference cache/events and optional redacted live text
phase2_results/            audit, protected-file hashes, tests and final report
```

The balanced LLM candidate file contains 80 unchanged English posts selected from
the ML dataset (shared post_key becomes post_id), plus 80 Hindi and 80 Hinglish
posts in `language_supplements_v3.jsonl`. The 160 additions are necessary because
the ML source has English text only. Twenty parallel scenarios each have four meal
contexts per language. They cover praise/inquiry, delays, refunds, safety/allergy,
privacy/security, sarcasm, ambiguity, multiple risks and injection. They are
correlated authored examples, not independent real-world complaints. All references
are provisional. English source labels remain partial: no invented urgency/safety
labels are filled in, and generic neutral posts have no asserted topic. All missing
fields require annotation. Report denominators therefore differ by field.

A deterministic scenario-family hash reserves held-out examples before inference;
Hindi/Hinglish meal variants of each family stay in the same partition. The prompt
is frozen before evaluation. Do not refine it against held-out results. Existing
English posts are also templated, so the split is not proof of real-world language
or semantic generalization. Full-sample results are descriptive; held-out and
development scores are available separately. No labels are changed to improve scores.

LLM Evaluation provides four performance/graph/error/playground tabs and a Human
annotation tab. Reviewers correct every canonical field, record their identifier,
notes and annotation version, then explicitly approve. Independent review hides
provisional labels and never reveals predictions or previous reviews. Latest
approved references can be evaluated; conflicting reviewers are excluded pending
adjudication. Explanatory wording is excluded from disagreement checks. Agreement
and Cohen's kappa require at least two overlapping reviewed posts. No human primary
accuracy is calculated until approved references exist and a report is refreshed.

Run or resume live evaluation:

```
python -m llm.evaluation --directory data/llm_evaluation/runs/my_gptoss_run --delay 5
python -m llm.evaluation --directory data/llm_evaluation/runs/my_gptoss_run --summarize-only
python -m llm.evaluate
python -m pytest -q
```

Successful checkpoints are reused; transport failures can be retried by resuming. Schema failures stay frozen; they are not repeatedly sampled until a preferred classification appears. Attempt-history counts remain visible. Input,
model and prompt hash key the cache. Resume rejects changed model/prompt/dataset
settings. Every attempt remains in a separate inference event log. Checkpoints
append; reports use the latest attempt per post. Recomputed runs with the same
prompt/records are not new independent tests. The original offline evaluator is
preserved; its old live mode redirects users to the canonical evaluation workflow.
Seven live-reference rows not belonging to the original benchmark are quarantined
without modifying the original 57-row CSV. Predictions are never written as truth.

Reports contain accuracy, macro/weighted F1, per-class precision/recall, confusion
matrices, safety micro/per-flag precision and recall, sarcasm, language slices,
JSON-plus-schema success, API failures, average/P95 uncached call latency and all
errors. A failed call is INVALID for categorical metrics and a miss for safety
recall, rather than silently excluded. Critical safety recall requires critical
urgency plus every expected safety flag on provisionally critical cases. It is
not comparable to the legacy risk-only recall. Per-language comparison is based
on sentiment where references exist. Predictions and provisional/human/historical
metrics are never pooled into one score. All Plotly traces use ordinary Python
lists to avoid binary-array rendering incompatibilities in older Streamlit.

Windows verification uses Python 3.11 and targeted Python 3.12 checks. Existing ML
pickles were trained with scikit-learn 1.2.2; Python 3.12 checks use 1.4.2 and issue
version warnings, so retain the trained-model runtime for exact reproducibility.
No artifact was retrained to silence warnings.


## Phase 3: Qdrant policy retrieval and response drafts

Launch the same app with `streamlit run app.py`, then select **RAG & Response Lab**.
The existing ML/LLM pages and model remain. Reusable interfaces:
`src.rag.retrieve.retrieve_policies(text, classification, top_k=5)`,
`src.rag.generate.generate_response(...)`, `src.workflow.pipeline.process_post(post_id)`
and `process_custom_post(text, metadata=None)`. Classification still uses the same
canonical module; `llm.client.shared_client` factors its original official Groq
client construction so drafting and classification share one client factory.

The current model is `openai/gpt-oss-120b` for classification and drafting.
The original secrets file was found under `data/.streamlit/` and copied into the
previously absent root `.streamlit/secrets.toml`. Its old Llama model setting made
classification calls fail; only the new root copy's non-secret GROQ_MODEL value
was corrected to GPT-OSS as explicitly requested. Credential and collection
values and the original source file were preserved. Never paste keys into logs.

App configuration continues to use Streamlit secrets first, environment fallback.
Standalone default is environment plus root `.env`, without overwriting process
variables. CLI `--secrets-file` is an explicit opt-in that copies string settings
into process defaults only. The old AIAgent `.env` is never loaded. QDRANT_COLLECTION
is respected, defaulting to `tomato_policies_v1`. Keys, URLs, .env files and local
RAG/model/evaluation outputs remain ignored. No publisher or Phase 4 engine exists.

```
python scripts/ingest_policies.py --prepare-only
python scripts/ingest_policies.py --secrets-file .streamlit/secrets.toml
python scripts/evaluate_rag.py --secrets-file .streamlit/secrets.toml --limit 8
python -m pytest -q
streamlit run app.py
```

For environment-configured scripts omit `--secrets-file`. The first embedding
load downloads the official `BAAI/bge-m3` ONNX files (about 2.3 GB). It uses the
model's dense CLS embedding, normalized to unit length, with 1,024 dimensions and
8,192-token maximum length; no alternate encoder is substituted. Both documents
and queries use this same encoder. The tested Windows runtime is pinned to
onnxruntime 1.19.2 after a native crash in 1.31.1. CPU weight prepacking and graph
rewrites are disabled to limit extra memory copies; loading remains substantial.
An optional BGE_M3_ENDPOINT must attest the exact model and return valid 1,024-D
vectors. The model is cached per process, vectors by content/model/config hash.
For future model-revision changes set BGE_MODEL_REVISION and re-ingest into a
compatible versioned collection. The live ingestion report records the downloaded
source revision. No model weights are changed or trained here.

The actual source corpus is seven DOCX files, not seven invented category files.
They contain 54 ID-bearing sections: 44 policy sections across six source documents
and ten template sections in the template document. Another 60 JSONL templates
remain supporting records. Templates and customer posts are excluded from the
policy collection. All policies are proposed academic-simulation rules, not
approved corporate policies. Missing approval/effective/active metadata stays null.
The parser preserves P0/P1 severity ranges, exact policy IDs, source/version,
conditions and exceptions. MD/TXT/JSONL/CSV adapters are available for explicit
ID-bearing clauses; the actual corpus uses DOCX plus supporting CSV/JSONL and
Markdown hierarchy files. Supporting scenario/reference files are never blindly
indexed as policies.

Ingestion produces `rag_results/policy_audit.json`, derived `rag_chunks.jsonl`,
`ingestion.json` and ID crosswalk information without rewriting source documents.
Deterministic UUIDs make upsert repeatable. Dimensions/distance are checked before
reuse. Keyword payload indexes are created because Qdrant Cloud strict mode
requires them for filters. Only stale points explicitly owned by this ingester are
synchronized; other collections are untouched. Repeated live ingestion leaves
44 actual vectors. The configured collection is `tomato_policies_v1`.

Retrieval uses normalized dense cosine search, default top 5 and minimum score
0.35, without tuning that cutoff against the evaluation cases. Only owned policy
records with the expected model are eligible. Applicability checks require relevant
issue-handling passages; high-risk cases need at least two distinct passages.
Conditions/exceptions remain attached to evidence. Similarity is not proof of
applicability or authorized compensation. Proposed policies are usable only as
simulation drafting context; there is no operational approval assertion.

The source documents genuinely conflict: SM-102 proposes P2 acknowledgement
within 60 minutes and P3 within four staffed hours, while SM-SLA-01 proposes two
and eight hours respectively. If relevant conflicting sections are retrieved,
the workflow abstains and requires review. Explicit precedence rows are retained;
references to missing IDs such as AR-101 are reported, not invented or silently
mapped. No deadlines are promised to customers.

Generation uses versioned prompts under `prompts/generation_versions/` (current 1.2.0; prior versions retained) and
Pydantic Draft/Support schemas. Exact retrieved policy IDs, exact evidence quotes
and exact draft claim spans are checked independently. Refund/compensation,
unverified case-status, private-detail, sensitive-information request and injection
patterns are blocked conservatively, including Hindi/Hinglish cases. High-risk
text and canonical safety flags force human escalation independently of ML spread
and even when a later component fails. Every result has human approval required
and publish_allowed=false. A nonblocked result is **review_required**, never
approved or published. Only those drafts get a copyable UI code block; blocked
outputs are suppressed and privacy-minimized in logs.

Deterministic guards are fallible proxies, not semantic entailment or clinical,
legal, translation or privacy certification. Conservative commitment patterns
can overblock safe language; exact quotes can reject paraphrases. The human rubric
covers policy support/conditions, citations, tone, language, safety, privacy and
injection. There are no fabricated subjective-quality scores or LLM-judge claims.

Evaluation references combine four unchanged ML posts, existing policy scenarios
and existing Phase 2 multilingual cases. All initial mappings are provisional.
The Reference review tab lets a person correct IDs, record reviewer/time/notes and
explicitly approve mappings, separately from predictions. Primary retrieval-quality
metrics remain pending until approved references exist. Provisional Recall@3/5,
Precision@5, MRR, latency/failures and response validation counts are computed from
actual checkpointed live results. Precision@5 uses denominator five, including
missing slots; sparse single-ID mappings are incomplete and require review.
Failures count as misses. Resume preserves completed outcomes and checks a model,
prompt, post-ID and reference manifest; it does not rerun outputs until they pass.
Separate run directories preserve configuration failures and later corrected runs.

Actual test reports, first configuration failure, corrected eight-post evaluation,
separate four-case safety/injection challenge and final evidence are under
`rag_results/`. Main eight-post provisional Recall@5 is 0.625 and MRR is 0.3542;
human metrics remain pending. Four representative ML posts all used the saved
model and its unchanged 0.22 threshold, while custom multilingual cases correctly
had ML unavailable. See `rag_results/PHASE3_REPORT.md` for complete observed results,
limits, file changes and verified commands.


The multilingual follow-up exposed Romanized Hinglish being answered in Hindi.
Generation prompt 1.2.0 uses a server-side writing-system preference without
changing the canonical classifier's language label. It also forbids future staff
investigation promises, and current guards block that wording in Hinglish. Old
outputs remain immutable; `python scripts/revalidate_drafts.py` adds separate
current-guard checks to saved reports without calling a model. Drafting currently
uses GPT-OSS low reasoning effort and a configurable 1,600-token output cap
(GROQ_DRAFT_MAX_TOKENS), with slower bounded backoff for rate limits. See the
[Groq reasoning documentation](https://console.groq.com/docs/reasoning) and
[official BGE-M3 files](https://huggingface.co/BAAI/bge-m3/tree/main/onnx).

A daily Groq token limit blocked some full-size follow-up requests. Those failures
remain visible, even though small requests and Qdrant work. No key was changed,
no fallback model was substituted, and no successful response was invented.
`--retry-transport` optionally retries only transport/API failures in a matching
saved evaluation run, retaining earlier attempts in the checkpoint. It does not
resample blocked/invalid drafts. A changed prompt/model/decoding configuration
requires a new run directory. The unreviewed main eight-post retrieval baseline
and separate safety/multilingual runs must not be described as human accuracy or
fresh independent test performance.

The default `streamlit` command in this Windows workspace resolves to Python 3.11,
which is the tested Phase 3 runtime. Python 3.12 does not yet have the new RAG
packages installed; the older Phase 2 targeted checks do not constitute a Phase 3
Python 3.12 verification. Do not retrain model artifacts to mask this limitation.


## Phase 4 — deterministic decisions and human review

Run `streamlit run app.py` and select **Crisis Monitoring & Reviewer Queue**. One versioned Python engine handles safety overrides, uncertain inputs, spread-based priority, routine drafts and monitoring. Missing ML remains unknown; the saved ML threshold and artifacts are unchanged. Decisions and explicit human review events persist in `data/review/decisions.sqlite3`; edits are versioned, unsafe approvals are blocked, and publication is disabled. The review identity is locally asserted, not an authentication system.

See [Phase 4 report](phase4_results/PHASE4_REPORT.md) for rule precedence, actual live examples, API-limit failures and verification. Run `python -m pytest -q --junitxml=phase4_results/tests_all.xml`. These rule tests are separate from model accuracy. Phases 5 and 6 are now integrated as described above.
