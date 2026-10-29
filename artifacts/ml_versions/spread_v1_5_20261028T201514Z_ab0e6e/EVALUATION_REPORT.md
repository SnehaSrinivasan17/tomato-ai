# Phase 1.5 final model report

{
  "model": "logistic_original_C1.0",
  "model_version": "spread_v1_5_20261028T201514Z_ab0e6e",
  "thresholds": {
    "default": 0.5,
    "f1_optimal": 0.22,
    "f2_optimal": 0.07,
    "operational": 0.22
  },
  "selection": "Largest validation average precision; exact ties by model name ascending.",
  "operational_rule": "Maximum validation F1 on 0.01\u20130.99 grid; exact ties choose higher threshold.",
  "trained_at_utc": "2026-10-28T20:18:22.467894+00:00",
  "phase": "1.5"
}

## Audit findings

No final outcome, reference labels or raw text in the predictor allowlist.
All authors are unique: author-purged temporal CV is defensive, not evidence of repeated-author robustness.
Post text/reference labels are sampled after engagement: text length/hashtags/mentions are deliberately weak controls, not semantic spread drivers.
Unobserved appeal, amplification, heavy-tailed late noise, rare shocks and Gamma-Poisson counts create overlap; no measured Bayes performance bound is claimed.
No verified historical-author engagement field exists; none is invented.
Prior cost-sensitive probability threshold is not F1-optimal. New threshold maximizes validation F1.
Original and confirmation tests were inspected in prior phases; this run cannot honestly call them untouched.
Generator and dataset are retained: changing noise after seeing prior tests would compromise evaluation.

## Training-only noise analysis

{
  "log_noise_variance": {
    "platform_amplification": 0.6553946023956861,
    "late_noise": 1.3235837718462482,
    "shock_multiplier": 0.24474481893488417
  },
  "latent_appeal_log_variance": 0.24733226280931533,
  "shock_prevalence": 0.011666666666666667,
  "log_feature_final_correlations": {
    "followers_count": 0.31095284794935507,
    "first_hour_impressions": 0.4610621552596723,
    "first_hour_likes": 0.4933654473370746,
    "first_hour_retweets": 0.5574727236224196
  },
  "retweet_quantiles": {
    "0.5": 4.0,
    "0.9": 43.0,
    "0.95": 91.0,
    "0.99": 557.0,
    "0.999": 5457.114000000023
  },
  "predictor_ap": {
    "followers_count": 0.22234249623961522,
    "first_hour_retweets": 0.37173033398266314,
    "first_hour_likes": 0.3705614229309163
  }
}

## Validation candidate comparison (offline only)

model,accuracy,precision,recall,f1,f2,balanced_accuracy,roc_auc,average_precision,brier,tn,fp,fn,tp,prevalence,alert_rate
logistic_original_C1.0,0.9063333333333333,0.75,0.17034700315457413,0.27763496143958866,0.20149253731343286,0.5818190476078499,0.7592882396582759,0.40449825238294335,0.07862671786162552,2665,18,263,54,0.10566666666666667,0.024
logistic_original_C0.1,0.907,0.7714285714285715,0.17034700315457413,0.27906976744186046,0.20179372197309417,0.5821917647155651,0.7598937579878449,0.40395720357551324,0.07866001122284041,2667,16,263,54,0.10566666666666667,0.023333333333333334
logistic,0.9063333333333333,0.7647058823529411,0.1640378548895899,0.2701298701298701,0.19461077844311378,0.579037190583073,0.7642746537081825,0.4022989028978911,0.07876540874383658,2667,16,265,52,0.10566666666666667,0.02266666666666667
random_forest,0.905,0.7424242424242424,0.15457413249211358,0.2558746736292428,0.18365817091454276,0.5741189708304772,0.7576715645065143,0.4020833125313385,0.07907719516060115,2666,17,268,49,0.10566666666666667,0.022
xgboost,0.9073333333333333,0.7910447761194029,0.167192429022082,0.27604166666666663,0.19850187265917604,0.5809871947570343,0.7594693072752733,0.39891388014926243,0.07883769703799202,2669,14,264,53,0.10566666666666667,0.022333333333333334
lightgbm,0.9073333333333333,0.8,0.1640378548895899,0.27225130890052357,0.19504876219054765,0.5795962662446459,0.7577027222458028,0.39745634078333947,0.07896612096055104,2670,13,265,52,0.10566666666666667,0.021666666666666667
hist_gradient,0.9063333333333333,0.7368421052631579,0.17665615141955837,0.28498727735368956,0.20833333333333334,0.5846009046326268,0.7512354337568825,0.37605554681493264,0.08022155933474859,2663,20,261,56,0.10566666666666667,0.025333333333333333


## Final test metrics

{
  "default": {
    "accuracy": 0.9016666666666666,
    "precision": 0.6753246753246753,
    "recall": 0.16149068322981366,
    "f1": 0.26065162907268175,
    "f2": 0.19047619047619047,
    "balanced_accuracy": 0.5760776791802541,
    "roc_auc": 0.79049559558213,
    "average_precision": 0.4009816546069866,
    "brier": 0.08008163019782648,
    "tn": 2653,
    "fp": 25,
    "fn": 270,
    "tp": 52,
    "prevalence": 0.10733333333333334,
    "alert_rate": 0.025666666666666667
  },
  "f1_optimal": {
    "accuracy": 0.8733333333333333,
    "precision": 0.40993788819875776,
    "recall": 0.40993788819875776,
    "f1": 0.40993788819875776,
    "f2": 0.40993788819875776,
    "balanced_accuracy": 0.6694947095960182,
    "roc_auc": 0.79049559558213,
    "average_precision": 0.4009816546069866,
    "brier": 0.08008163019782648,
    "tn": 2488,
    "fp": 190,
    "fn": 190,
    "tp": 132,
    "prevalence": 0.10733333333333334,
    "alert_rate": 0.10733333333333334
  },
  "f2_optimal": {
    "accuracy": 0.657,
    "precision": 0.20566194837635304,
    "recall": 0.7670807453416149,
    "f1": 0.3243598161523309,
    "f2": 0.4961832061068703,
    "balanced_accuracy": 0.7054223741644594,
    "roc_auc": 0.79049559558213,
    "average_precision": 0.4009816546069866,
    "brier": 0.08008163019782648,
    "tn": 1724,
    "fp": 954,
    "fn": 75,
    "tp": 247,
    "prevalence": 0.10733333333333334,
    "alert_rate": 0.4003333333333333
  },
  "operational": {
    "accuracy": 0.8733333333333333,
    "precision": 0.40993788819875776,
    "recall": 0.40993788819875776,
    "f1": 0.40993788819875776,
    "f2": 0.40993788819875776,
    "balanced_accuracy": 0.6694947095960182,
    "roc_auc": 0.79049559558213,
    "average_precision": 0.4009816546069866,
    "brier": 0.08008163019782648,
    "tn": 2488,
    "fp": 190,
    "fn": 190,
    "tp": 132,
    "prevalence": 0.10733333333333334,
    "alert_rate": 0.10733333333333334
  },
  "calibration_ece": 0.009511014038026585
}

## Previous versus new: same original test

{
  "previous": {
    "directory": "artifacts\\synthetic_ml",
    "threshold": 0.18,
    "metrics": {
      "accuracy": 0.8556666666666667,
      "precision": 0.3669064748201439,
      "recall": 0.4751552795031056,
      "f1": 0.4140730717185385,
      "f2": 0.44868035190615835,
      "balanced_accuracy": 0.68828712444162,
      "roc_auc": 0.79049559558213,
      "average_precision": 0.4009816546069866,
      "brier": 0.08008163019782648,
      "tn": 2414,
      "fp": 264,
      "fn": 169,
      "tp": 153,
      "prevalence": 0.10733333333333334,
      "alert_rate": 0.139
    }
  },
  "new": {
    "version": "spread_v1_5_20261028T201514Z_ab0e6e",
    "threshold": 0.22,
    "metrics": {
      "accuracy": 0.8733333333333333,
      "precision": 0.40993788819875776,
      "recall": 0.40993788819875776,
      "f1": 0.40993788819875776,
      "f2": 0.40993788819875776,
      "balanced_accuracy": 0.6694947095960182,
      "roc_auc": 0.79049559558213,
      "average_precision": 0.4009816546069866,
      "brier": 0.08008163019782648,
      "tn": 2488,
      "fp": 190,
      "fn": 190,
      "tp": 132,
      "prevalence": 0.10733333333333334,
      "alert_rate": 0.10733333333333334
    }
  },
  "comparable": true,
  "conditions": "Same original test posts, labels and target; operational thresholds may differ. Test was already inspected in earlier phases."
}

## Interpretation

No Bayes-optimal performance ceiling has been established. Predictive early engagement exists, but unobserved late noise, amplification and rare shocks create unavoidable overlap for these inputs. The generator was not edited.
Search train-to-CV AP gaps diagnose overfitting; calibration ECE and Brier quantify probability quality. The final test is reused from earlier phases, so these results are not a claim of a newly untouched evaluation.
F1 threshold selection is validation-only. Dashboard test-threshold exploration is descriptive and never writes model thresholds.
Exact counts, distributions, feature order, data hashes, fitting parameters and curve arrays are in this immutable bundle.