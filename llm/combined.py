"""Independent LLM and existing ML inference, followed by illustrative triage."""
import math

from llm.classifier import classify_post
from ml.inference import SpreadPredictor


def review_decision(llm_output, spread=None,post_id="combined_analysis"):
    """Compatibility projection of the ONE deterministic engine, not another rule set."""
    from src.decision.engine import decide
    d=decide(post_id,llm_output,spread)
    labels=llm_output or {}
    return dict(high_severity_flag=d.action.value=='URGENT_ESCALATION' or 'B_SERIOUS' in d.rule_ids,
        negative_sentiment=labels.get('sentiment')=='negative',spread_alert=d.high_spread,
        priority_review=d.action.value in {'PRIORITY_REVIEW','URGENT_ESCALATION'},
        human_review_required=d.requires_human_review,decision=d.model_dump(mode='json'))


def validate_metadata(metadata, schema):
    """Require every ML feature; zero is valid, missing is not imputed here."""
    if not isinstance(metadata, dict):
        return {}, ['Metadata must be a mapping of all required ML fields.']
    cleaned, errors = {}, []
    for feature in schema['features']:
        raw = metadata.get(feature)
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            errors.append(feature + ': required')
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            errors.append(feature + ': numeric value required')
            continue
        bounds = schema.get('bounded', {}).get(feature)
        if not math.isfinite(value) or value < 0 or value != int(value) or (bounds and not bounds[0] <= value <= bounds[1]):
            errors.append(feature + ': nonnegative integer within allowed bounds required')
        else:
            cleaned[feature] = value
    if not errors:
        for name in ['first_hour_likes', 'first_hour_replies', 'first_hour_retweets']:
            if name in cleaned and cleaned[name] > cleaned['first_hour_impressions']:
                errors.append(name + ': cannot exceed first-hour impressions')
    return cleaned, errors


def combined_analysis(post_text, metadata=None, *, classify=None, predictor=None):
    classifier = classify or classify_post
    result = classifier(post_text)
    if isinstance(result, tuple):  # injected legacy callbacks only
        llm_ok, llm_output, llm_error = result
    else:
        llm_ok = result.status == 'success'
        llm_output = result.classification.model_dump(mode='json') if llm_ok else None
        llm_error = result.error_message
    post_id=getattr(result,'post_id','combined_analysis')
    spread, metadata_errors = None, []
    try:
        predictor = predictor or SpreadPredictor()
        values, metadata_errors = validate_metadata(metadata or {}, predictor.schema)
        if not metadata_errors:
            spread = predictor.predict(values)
    except (OSError, ValueError, KeyError):
        metadata_errors = ['The existing ML model could not be loaded.']
    return dict(llm=llm_output if llm_ok else None, llm_error=llm_error,
                ml=spread, metadata_errors=metadata_errors,
                review=review_decision(llm_output if llm_ok else None,spread,post_id))
