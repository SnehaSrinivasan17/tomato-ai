import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                            confusion_matrix, fbeta_score, precision_score, recall_score, roc_auc_score)
from ml.data import FEATURES


def prepare_input(values, features=None, bounds=None):
    """Accept partial metadata; missing fields use fitted training medians."""
    features = FEATURES if features is None else features
    frame = pd.DataFrame([values] if isinstance(values, dict) else values).reindex(columns=features)
    for name in features:
        frame[name] = pd.to_numeric(frame[name], errors='coerce')
        frame.loc[~np.isfinite(frame[name]) | (frame[name] < 0), name] = np.nan
    for name, maximum in [('hour_utc', 23), ('weekday', 6), ('verified', 1), ('is_reply', 1), ('is_quote', 1)]:
        if name in frame:
            frame.loc[frame[name] > maximum, name] = np.nan
    for name, (_, maximum) in (bounds or {}).items():
        if name in frame:
            frame.loc[frame[name] > maximum, name] = np.nan
    return frame


def metrics(y, probability, threshold=.5):
    predicted = np.asarray(probability) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    both = len(set(y)) == 2
    return dict(accuracy=float(accuracy_score(y, predicted)), precision=float(precision_score(y, predicted, zero_division=0)),
                recall=float(recall_score(y, predicted, zero_division=0)),
                f1=float(fbeta_score(y, predicted, beta=1, zero_division=0)),
                f2=float(fbeta_score(y, predicted, beta=2, zero_division=0)),
                balanced_accuracy=float(balanced_accuracy_score(y, predicted)),
                roc_auc=float(roc_auc_score(y, probability)) if both else None,
                average_precision=float(average_precision_score(y, probability)) if both else None,
                brier=float(brier_score_loss(y, probability)),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                prevalence=float(np.mean(y)), alert_rate=float(np.mean(predicted)))
