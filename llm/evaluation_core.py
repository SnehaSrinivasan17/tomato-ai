"""Tomato Social Listening - Automated Evaluation System
Author: Tomato AI Engineering
Description:
    Evaluates LLM performance (Groq API) against 50 synthetic Tomato
    customer posts across 8 classification dimensions.
    Calculates field-by-field accuracy, macro precision, recall, F1,
    JSON schema compliance, latency, and critical food-safety/privacy recall.
"""

import os
import sys
import csv
import json
import time
import math
from datetime import datetime
from collections import defaultdict

VALID_SENTIMENTS = {"positive", "negative", "neutral"}
VALID_TOPICS = {"food_quality", "delivery", "app_service", "billing", "customer_support", "general_inquiry"}
VALID_URGENCIES = {"low", "medium", "high", "critical"}
VALID_RISK_FLAGS = {"none", "food_safety", "privacy", "harassment", "legal"}

EXPECTED_SCHEMA_KEYS = [
    "sentiment", "topic", "urgency", "needs_reply",
    "risk_flag", "language", "sarcasm", "human_review"
]


def load_classifier_prompt(prompt_path: str) -> str:
    """Load system prompt from file."""
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Classifier prompt not found at: {prompt_path}")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_dataset(dataset_path: str):
    """Load synthetic posts dataset with reference labels."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")
    posts = []
    with open(dataset_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            posts.append({
                "post_id": row["post_id"].strip(),
                "post_text": row["post_text"].strip(),
                "sentiment": row["sentiment"].strip().lower(),
                "topic": row["topic"].strip().lower(),
                "urgency": row["urgency"].strip().lower(),
                "needs_reply": row["needs_reply"].strip().lower() in ("true", "1", "yes"),
                "risk_flag": row["risk_flag"].strip().lower(),
                "language": row["language"].strip().lower(),
                "sarcasm": row["sarcasm"].strip().lower() in ("true", "1", "yes"),
                "human_review": row["human_review"].strip().lower() in ("true", "1", "yes"),
            })
    return posts


def validate_prediction_schema(pred_obj):
    """
    Validate prediction object against Tomato schema.
    Returns: (is_valid, normalized_pred, validation_errors)
    """
    errors = []
    normalized = {}

    if not isinstance(pred_obj, dict):
        return False, {}, ["Model output was not a valid dictionary"]

    # Sentiment
    sent = str(pred_obj.get("sentiment", "")).strip().lower()
    if sent not in VALID_SENTIMENTS:
        errors.append(f"Invalid sentiment: '{sent}'")
    normalized["sentiment"] = sent if sent in VALID_SENTIMENTS else "unknown"

    # Topic
    top = str(pred_obj.get("topic", "")).strip().lower()
    if top not in VALID_TOPICS:
        errors.append(f"Invalid topic: '{top}'")
    normalized["topic"] = top if top in VALID_TOPICS else "unknown"

    # Urgency
    urg = str(pred_obj.get("urgency", "")).strip().lower()
    if urg not in VALID_URGENCIES:
        errors.append(f"Invalid urgency: '{urg}'")
    normalized["urgency"] = urg if urg in VALID_URGENCIES else "unknown"

    # Needs reply
    nr = pred_obj.get("needs_reply")
    if isinstance(nr, bool):
        normalized["needs_reply"] = nr
    elif str(nr).lower() in ("true", "1", "yes"):
        normalized["needs_reply"] = True
    elif str(nr).lower() in ("false", "0", "no"):
        normalized["needs_reply"] = False
    else:
        errors.append(f"Invalid needs_reply: '{nr}'")
        normalized["needs_reply"] = False

    # Risk flag
    rf = str(pred_obj.get("risk_flag", "")).strip().lower()
    if rf not in VALID_RISK_FLAGS:
        errors.append(f"Invalid risk_flag: '{rf}'")
    normalized["risk_flag"] = rf if rf in VALID_RISK_FLAGS else "unknown"

    # Language
    lang = str(pred_obj.get("language", "")).strip().lower()
    if not lang:
        errors.append("Empty language code")
    normalized["language"] = lang if lang else "unknown"

    # Sarcasm
    sc = pred_obj.get("sarcasm")
    if isinstance(sc, bool):
        normalized["sarcasm"] = sc
    elif str(sc).lower() in ("true", "1", "yes"):
        normalized["sarcasm"] = True
    elif str(sc).lower() in ("false", "0", "no"):
        normalized["sarcasm"] = False
    else:
        errors.append(f"Invalid sarcasm: '{sc}'")
        normalized["sarcasm"] = False

    # Human review
    hr = pred_obj.get("human_review")
    if isinstance(hr, bool):
        normalized["human_review"] = hr
    elif str(hr).lower() in ("true", "1", "yes"):
        normalized["human_review"] = True
    elif str(hr).lower() in ("false", "0", "no"):
        normalized["human_review"] = False
    else:
        errors.append(f"Invalid human_review: '{hr}'")
        normalized["human_review"] = False

    is_valid = len(errors) == 0
    return is_valid, normalized, errors


def compute_categorical_metrics(references, predictions, possible_classes=None):
    """Calculate accuracy, macro precision, recall, F1 for categorical labels."""
    total = len(references)
    if total == 0:
        return 0.0, 0.0, 0.0, 0.0, 0, 0

    correct = sum(1 for r, p in zip(references, predictions) if r == p)
    accuracy = correct / total

    classes = set(references) | set(predictions)
    if possible_classes:
        classes = classes | set(possible_classes)
    # Remove 'unknown' or empty from macro averages if undesired
    classes = [c for c in classes if c not in ("unknown", "")]

    if not classes:
        return accuracy, 0.0, 0.0, 0.0, total, correct

    precisions = []
    recalls = []
    f1s = []

    for c in classes:
        tp = sum(1 for r, p in zip(references, predictions) if r == c and p == c)
        fp = sum(1 for r, p in zip(references, predictions) if r != c and p == c)
        fn = sum(1 for r, p in zip(references, predictions) if r == c and p != c)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    macro_precision = sum(precisions) / len(precisions) if precisions else 0.0
    macro_recall = sum(recalls) / len(recalls) if recalls else 0.0
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    return accuracy, macro_precision, macro_recall, macro_f1, total, correct


def compute_boolean_metrics(references, predictions):
    """Calculate accuracy, positive precision, positive recall, positive F1."""
    total = len(references)
    if total == 0:
        return 0.0, 0.0, 0.0, 0.0, 0, 0

    correct = sum(1 for r, p in zip(references, predictions) if bool(r) == bool(p))
    accuracy = correct / total

    tp = sum(1 for r, p in zip(references, predictions) if bool(r) is True and bool(p) is True)
    fp = sum(1 for r, p in zip(references, predictions) if bool(r) is False and bool(p) is True)
    fn = sum(1 for r, p in zip(references, predictions) if bool(r) is True and bool(p) is False)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return accuracy, prec, rec, f1, total, correct


def build_confusion_matrix(references, predictions, classes):
    """Build a square confusion matrix dict [ref][pred]."""
    matrix = {r: {p: 0 for p in classes} for r in classes}
    for ref, pred in zip(references, predictions):
        r_key = ref if ref in classes else "other"
        p_key = pred if pred in classes else "other"
        if r_key in matrix and p_key in matrix[r_key]:
            matrix[r_key][p_key] += 1
    return matrix


