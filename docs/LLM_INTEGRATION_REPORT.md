# LLM integration verification · 2026-10-29

- Python 3.11 full suite: 39 passed.
- Python 3.12 full suite: 38 passed, 1 archived diagnostic-model compatibility failure.
- Python 3.12 focused integration suite: 20 passed.
- Original AIAgent read-only validation script: 7/7 checks passed.
- Actual Groq call: succeeded on a fictional food-safety post; critical urgency, food_safety risk and human review. No persistence.
- Offline benchmark recomputation: matched all eight saved metric rows and 25 saved field errors.
- Historical/ML preservation: 133 ML source/data/artifact/report files and 28 original AIAgent files verified unchanged. Original prompt is byte-identical.
- Primary model numerical compatibility under Python 3.12: all 3,000 original held-out probabilities match with maximum absolute difference 2.220446049250313e-16.
- Streamlit HTTP health: 200. All four navigation sections passed AppTest. Missing keys, bad text/JSON, missing metadata, missing model and high-severity escalation were tested.

## Failure retained transparently

The one Python 3.12 failure is tests/test_ml.py::PipelineTests::test_artifacts_and_probabilities, which explicitly loads artifacts/repost_diagnostic/model.joblib. Its archived Random Forest was serialized by scikit-learn 1.2.2; the Python 3.12 runtime uses 1.4.2, whose tree node layout differs. The integrated app uses the unchanged synthetic Logistic Regression instead. No model was retrained or replaced to hide this failure. The old diagnostic still passes in the original Python 3.11 environment.

## Scope

The original interactive legacy script was not executed against source data because it appends predictions into reference-label CSVs. Equivalent integration tests use temporary live files; benchmark and live records are isolated. Imported benchmark statistics remain source evaluations, not a claimed new 50-post Groq run.

## Launch

```powershell
streamlit run app.py
```

For tested Python 3.12 versions: py -3.12 -m pip install -r requirements.txt -c constraints-python312.txt, then py -3.12 -m streamlit run app.py. The root .env is ignored; no credentials appear in this report.
