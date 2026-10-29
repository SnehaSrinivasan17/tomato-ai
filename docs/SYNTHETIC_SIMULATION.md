# Fictional food-delivery spread simulation v1

This educational simulator is specified independently of model scores. No search
over generator parameters, performance-target tuning, or rejection sampling is
performed. All reported metrics come from training and held-out predictions.
The parameter registry is `PARAMETERS` in `ml/synthetic.py`, persisted alongside
the datasets in `simulation_manifest.json`. Seed 20261029 generates 15,000 main
posts; seed 20261030 generates 3,000 separate shifted test posts.

## Generative process

Every post has a unique fictional author. This deliberate simplification prevents
author leakage but does not recreate repeated-customer dynamics. Dates are sampled
uniformly across 90 fictional days beginning January 1, 2026. Followers follow a
lognormal distribution (log-mean 5.7, log-standard-deviation 1.75), capped at two
million for numerical safety. Following and lifetime post counters are independent
lognormal draws; list membership and verified probability depend on follower count.
These are fictional metadata, not calibrated estimates of real social-media users.

Let F be followers, M the media flag, A ~ Normal(0,1) unobserved appeal, and
E ~ LogNormal(-0.25,0.75) independent early exposure noise. Expected first-hour
impressions are:

`reach_scale * (4 + 1.5*(F+1)^0.66) * exp(0.45*A + media_effect*M) * E`.

Actual impressions are Poisson draws, capped in the mean at five million.
First-hour likes, replies and retweets are binomial draws from impressions with
probabilities `0.035*exp(0.25*A)`, `0.004*exp(0.2*A)` and
`0.008*exp(0.3*A)`, respectively, clipped to the safe ranges in source.
These actions can overlap, as a person may perform several actions.

Let P be independent platform amplification from
LogNormal(-sigma²/2,sigma), L be `exp(clip(t4*late_noise,-5,6))`, and S be a
rare shock. Independently for each post, S=1 normally; with `shock_probability`,
S=`1 + shock_scale*(1+Pareto(2.2))`. The mean additional retweets is:

`(2 + 1.8*early_retweets + 0.25*(early_likes+1)^0.8 + 0.15*(F+1)^0.35)`
`* exp(0.5*A + media_effect*M) * P * L * S`.

Additional retweets are sampled using a Gamma(shape=2, scale=mean/2)–Poisson
mixture; its mean is capped at ten million only to avoid numerical extremes.
Final retweets equal first-hour retweets plus this additional stochastic draw.
Heavy-tailed reach, amplification, Student-t noise and Pareto shocks create
overlap, missed viral cases, and occasional small-account viral posts. No account
size is guaranteed to be ordinary or viral. Even knowing every ML input does not
reveal later platform draws, shocks, latent appeal or count noise.

## Text and content metadata

Posts use fictional restaurants and food items, order amounts, delays, and
sentence variants. Eight reference topics and positive/negative/neutral template
families are retained for later LLM evaluation and integrated demos. Hashtags,
mentions and reserved `.invalid` URLs are sampled; media is a separate attachment
flag. Length, hashtag/mention counts, link flag and media flag are explicit content
metadata. Media is a direct generative driver; the other content features are
deliberately weak or irrelevant controls rather than guaranteed predictors.
Templates do not establish real language diversity or human-labelled ground truth.
No real personal information, real order identifiers, or API calls are used.

## Exclusions and prediction time

The ML whitelist consists of author counters, verification, age, UTC hour/weekday,
reply/quote flags, numeric content metadata and four first-hour engagement counts.
Raw text, sentiment/topic reference labels, urgency, final engagement, identities,
regime markers and latent simulation variables are excluded. Hidden draws are
saved in separate audit CSVs, never joined into model inputs. Final retweets are
used only to construct labels. This is an **after-first-hour prediction of eventual
synthetic spread**, not a prediction available at publication or a forecast of
first-hour engagement. Labels describe spread, not safety or crisis severity.

## Splits, threshold and evaluation

Chronologically sort the main data and split near 60%/20%/20%, keeping tied
timestamps together. Only then calculate the training 90th percentile using
linear interpolation. A post is positive when final retweets **>=** this saved
number. All boundary ties are included, so prevalence can differ from exactly 10%.
Both held-out sets and the shifted set use that unchanged target definition.
No test-set percentile is computed to redefine positives.

Fit median imputation, log1p transform and scaling on training data only.
Compare a dummy baseline and two configurations each of logistic regression,
random forest and histogram gradient boosting. Choose highest validation average
precision (reported PR-AUC summary); model name breaks ties. Sweep probabilities
0.05–0.95 in 0.01 increments on validation data only. Operational cost is
`5*FN + FP`, breaking ties toward lower thresholds. Also record 0.50, F1-optimal
and F2-optimal thresholds. Freeze the pipeline and thresholds before final test
and shifted-test prediction. Neither test is used for model or generator tuning.
Report Brier and reliability bins without post-hoc test calibration. Calibration
bins are equal-width and empty bins are omitted. Feature importance uses repeated
validation permutation scored by average precision. It is not causal attribution.

## Shifted simulation

| Parameter | Main distribution | Shifted test |
|---|---:|---:|
| Reach scale | 1.0 | 0.65 |
| Media effect | 0.35 | 0.05 |
| Late-noise multiplier | 0.85 | 1.2 |
| Amplification log sigma | 0.8 | 1.15 |
| Rare-shock probability | 0.015 | 0.04 |
| Shock scale | 60 | 95 |

The shifted dataset includes both changed input distributions and changed outcome
mechanisms. Its base rate may differ; operational thresholds remain frozen.
This single stress scenario is not a comprehensive robustness certification.
Strong synthetic results cannot establish real-world accuracy, calibration,
Indian location effects, or crisis suitability. Validate on representative,
consented, timestamped real data before operational use. Historical originals'
degenerate-label report remains accessible in the dashboard; neither the historical
originals nor repost diagnostic models are primary inference fallbacks.
