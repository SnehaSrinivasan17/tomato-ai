# Synthetic primary ML validation report

All outcomes and metrics concern fictional simulation data only. Generator v1 was fixed independently of model scores. No historical repost model is used for primary inference.

Selected model: logistic_C1.0
Training 90th-percentile cutoff: 43.0
Operational threshold: 0.18 (validation 5*FN+FP minimum)

| Metric | In-distribution test | Shifted simulation test |
|---|---:|---:|
| average_precision | 0.400982 | 0.220154 |
| roc_auc | 0.790496 | 0.628090 |
| precision | 0.366906 | 0.250980 |
| recall | 0.475155 | 0.196319 |
| f1 | 0.414073 | 0.220310 |
| f2 | 0.448680 | 0.205260 |
| brier | 0.080082 | 0.095021 |
| tn | 2414.000000 | 2483.000000 |
| fp | 264.000000 | 191.000000 |
| fn | 169.000000 | 262.000000 |
| tp | 153.000000 | 64.000000 |
| prevalence | 0.107333 | 0.108667 |

## Split counts and prevalence

- train: 9000 posts; 910 positives (10.111111%); 2026-01-01 00:33:16+00:00 to 2026-02-24 08:31:51+00:00
- validation: 3000 posts; 317 positives (10.566667%); 2026-02-24 08:32:49+00:00 to 2026-03-14 02:07:34+00:00
- test: 3000 posts; 322 positives (10.733333%); 2026-03-14 02:11:45+00:00 to 2026-03-31 23:44:29+00:00
- shifted_test: 3000 posts; 326 positives (10.866667%); 2026-01-01 00:58:16+00:00 to 2026-03-31 23:13:21+00:00

Small-account high-spread cases (<500 followers): 590.

## Validation

Full suite: 14 passed. AppTest verified predictions, reset, threshold exploration, historical-report access, and supported Arrow table payloads. Independent retraining reproduced metrics, selection, labels, defaults and manifests; probabilities matched within 1e-14. Local Python 3.11 was used; Python 3.12 was not tested. Legacy historical models still emit sklearn/pandas deprecation warnings.

## Limitations

- All posts, authors, engagement and reference labels are fictional simulation outputs, not observed customer data.
- Scores measure recovery of assumptions in this simulator, not performance on real social media or Indian customers.
- Prediction is made after the first hour; it classifies eventual simulated spread, not a forecast of first-hour engagement.
- Independent latent appeal, platform amplification, late noise and rare shocks remain unobserved by ML.
- The shifted test is one specified stress scenario, not comprehensive evidence of real-world robustness.
- Reference sentiment/topic labels come from templates and are not human-annotated ground truth.
- The 5:1 missed-positive to false-alert cost is a prototype assumption; all responses still need human review.
