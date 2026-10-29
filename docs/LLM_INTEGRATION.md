# Existing LLM classifier integration

## Preserved components

`AIAgent/` remains intact. `llm/classifier.py` adapts its existing `classify_post`
implementation; `llm/prompts/classifier_prompt.txt` is byte-identical to the
source prompt. Default Groq model: `openai/gpt-oss-120b`, temperature 0, JSON mode.
`GROQ_MODEL` remains an explicit configuration override. No new prompt, RAG, or
response generator is introduced.

The eight classification fields remain sentiment, topic, urgency, needs_reply,
risk_flag, language, sarcasm and human_review. Additional original outputs remain
available: confidence_score, confidence_pct, human_review_reason, language_display,
latency_ms, timestamps and live post identifiers. Missing confidence stays null
rather than being replaced by the source app's invented 0.94 value. Confidence
is model-reported, not an empirically calibrated probability.

The integration fixes string-boolean parsing, rejects malformed/incomplete JSON,
handles invalid/empty text, and sanitizes SDK exception messages. The prompt and
classification taxonomy are unchanged. Groq clients use a 45-second timeout and
two SDK retries. Live IDs keep the `TOM_LIVE_` prefix and use UUIDs for uniqueness.
Persistence is opt-in, serialized within the Streamlit process, deduplicated and
backed up locally. Multi-process writers require an external database/lock.

## Record separation and evaluation

Source files contain 57 rows: 50 evaluated benchmark posts and seven live rows.
Benchmark IDs are taken from the existing `comparison_results.csv`, not from
row position. Only those IDs enter the integrated benchmark reference and
prediction CSVs. Source metric/error/comparison/summary files are preserved;
raw imported files are retained in a private source snapshot.

The old persistence code copied predicted labels into the reference dataset.
Imported live reference rows are quarantined as unverified data, not ground truth.
New live submissions never write benchmark or reference-label files. Saving is
off by default and uses `data/llm/live/groq_predictions.csv` with local backups.
Combined Analysis does not persist submissions. All these data directories are
Git-ignored. No live text enters ML training, benchmark scoring, or model logs.

`llm/evaluation_core.py` reuses the source evaluator's metric implementations.
Its boolean-field precision/recall/F1 measure the positive class, even though the
original CSV column names say `macro_*`; categorical fields use macro averages.
The dashboard reads the existing field metrics and error analysis dynamically;
confusion matrices are recomputed by joining saved benchmark predictions to
references by post ID. Existing benchmark results are not presented as a fresh run.
Reference labels are synthetic draft labels, not verified operational ground truth.

```powershell
python -m llm.evaluate
python -m llm.classifier --post "My fictional order arrived cold."
```

Offline evaluation writes a new timestamped directory under
`data/llm/evaluations/`. To explicitly rerun the benchmark against Groq, use
`python -m llm.evaluate --live`. It uses the same classifier and writes a new
evaluation directory, never overwriting the imported benchmark or ML artifacts.
Network failures abort the new run instead of silently dropping failed records.

## Independent combined inference

The combined page collects text and every numeric input in the existing ML
feature schema. All metadata fields begin blank. Missing, nonnumeric, out-of-range,
or inconsistent first-hour inputs yield no ML probability. Existing ML-only
Try It Yourself defaults and inference behavior are preserved unchanged.

The LLM receives only the post text. The existing `SpreadPredictor` receives only
the supplied author/content/first-hour metadata. Topic, sentiment and urgency do
not become ML inputs. The saved training virality definition and operational
probability threshold are read from the existing artifacts.

The illustrative routing rule is:

`priority_review = high_severity_flag OR (negative_sentiment AND spread_alert)`

High severity means a non-none risk flag or high/critical urgency. It requires
review even with low spread or missing metadata. The original LLM human_review
field remains visible, while combined human_review_required also enforces severe
case escalation. With negative sentiment and no ML prediction, priority stays
unknown rather than fabricating a false/zero spread result. If LLM classification
fails, priority is unknown and manual review is required; valid ML inference may
still be displayed independently. These are model-based flags, not guarantees
that the LLM detects every severe post. No reply is generated or published.

## Environment and security

Only the project root `.env` is loaded, with `override=False`. Existing process
variables are not replaced. During Streamlit execution, existing `st.secrets`
values take precedence over the environment. Standalone scripts use environment
variables plus the root `.env`. No Qdrant credential is required for classification.

If root `.env` was absent, the existing AIAgent Groq settings were copied into it
without changing the source. Existing root configuration is never overwritten.
Both source `.env` and root `.env` remain ignored; only root `.env` is used by the
integrated app. The classifier does not log API keys, raw provider errors, or post
text. User-entered text is sent to Groq when classification is explicitly run.

## Compatibility and verification

The full project suite passed on Python 3.11 (39 tests). The focused integration
suite also passed all 20 tests on Python 3.12.
The original read-only `AIAgent/test_app_validation.py` passed all seven checks.
An actual Groq call on a fictional food-safety post succeeded without persistence.
Offline recomputation matched all eight supplied metric rows and all 25 errors.
The prompt hash is unchanged. An audit snapshot covers all existing ML source,
data, model artifacts and reports; hashes are compared after integration.

Python 3.12.0 is installed at the Anaconda interpreter, using scikit-learn 1.4.2,
Streamlit 1.32.0 and pandas 2.2.2. The primary Logistic Regression loads and all
3,000 original test probabilities match the saved values within 2.23e-16. The
integrated pages and LLM tests pass there. A full-suite compatibility check found
one pre-existing archived repost-model failure: the old Random Forest pickle was
trained with scikit-learn 1.2.2 and cannot load its tree-node layout under 1.4.2.
It is not used by the app. It was neither replaced nor retrained. Cross-version
scikit-learn loading still emits warnings; the recorded original Python 3.11
environment remains necessary for that archived diagnostic. No ML retraining was
performed. Test logs and numerical parity evidence are in `integration_results/`.

The original interactive legacy script is not run verbatim because it appends
live rows and predicted labels into the original reference CSV, and its assertions
contradict that persistence behavior. Equivalent classifier/persistence/isolation
checks run against the integrated module with disposable files and injected API
responses; one additional actual live Groq smoke call verifies service operation.

For the tested Python 3.12 dependency combination, install with:

```powershell
py -3.12 -m pip install -r requirements.txt -c constraints-python312.txt
py -3.12 -m streamlit run app.py
```

With the intended Python environment active, the standard launch command remains
`streamlit run app.py`. `pytest.ini` scopes automatic collection to `tests/` so
legacy scripts with side effects are not inadvertently run against source data.
