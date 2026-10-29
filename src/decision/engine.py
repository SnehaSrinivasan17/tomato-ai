"""ONE deterministic action authority; no API calls, model training or publishing."""
import json,math,hashlib
from pathlib import Path
from llm.schemas import Classification
from src.rag.guardrails import risk_flags
from ml.bundle import active_model_directory,validate_bundle
from src.decision.schemas import Decision,Action,Priority
from src.decision.rules import VERSION,RULES,SERIOUS_FLAGS,IMMINENT,ACTIVE_EXPOSURE,IMMINENT_THREAT,DISCRIMINATION,POLICY_BINDINGS


RULE_FINGERPRINT=hashlib.sha256(b''.join((Path(__file__).parent/name).read_bytes() for name in ['engine.py','rules.py','schemas.py'])).hexdigest()

def frozen_threshold():
    directory=active_model_directory();validate_bundle(directory)
    return float(json.loads((directory/'selection.json').read_text())['thresholds']['operational'])


def validated_labels(value):
    try:
        if hasattr(value,'model_dump'):value=value.model_dump(mode='json')
        if isinstance(value,dict) and 'status' in value:
            if value['status']!='success':return None
            value=value.get('classification')
        return Classification.model_validate(value).model_dump(mode='json')
    except (ValueError,TypeError):return None


def decide(post_id,classification=None,spread=None,*,post_text='',retrieval=None,generation=None,stage='preliminary',response_required_by_context=False):
    if type(response_required_by_context) is not bool:raise ValueError('Response context must be an explicit server-side boolean.')
    if stage not in {'preliminary','post_retrieval','final'}:raise ValueError('Unknown decision stage.')
    labels=validated_labels(classification);text=post_text if isinstance(post_text,str) else ''
    missing=[];modifiers=[];reasons=[];p=None;threshold=None;high=None
    try:threshold=frozen_threshold()
    except (OSError,ValueError,KeyError,TypeError):missing.append('saved_operational_threshold')
    try:
        if not isinstance(spread,dict) or isinstance(spread.get('probability'),bool):raise ValueError()
        probability=float(spread['probability']);provided=float(spread['operational_threshold'])
        if not math.isfinite(probability) or not 0<=probability<=1 or threshold is None or not math.isclose(provided,threshold,abs_tol=1e-12):raise ValueError()
        signal=probability>=threshold
        if 'high_spread' in spread and (type(spread['high_spread']) is not bool or spread['high_spread']!=signal):raise ValueError()
        p=probability;high=signal
    except (ValueError,TypeError,KeyError):
        missing.append('ml_prediction');modifiers.append('F_ML_UNAVAILABLE');reasons.append(RULES['F_ML_UNAVAILABLE'])
    flags=set(labels['safety_flags']) if labels else set()
    if IMMINENT.search(text):flags.add('text_imminent_harm')
    if ACTIVE_EXPOSURE.search(text):flags.add('text_active_privacy_or_compromise')
    if IMMINENT_THREAT.search(text):flags.add('text_imminent_threat')
    if DISCRIMINATION.search(text):flags.add('text_discrimination')
    critical=bool(flags & {'medical_emergency','text_imminent_harm','text_active_privacy_or_compromise','text_imminent_threat'})
    critical=critical or bool(labels and flags & SERIOUS_FLAGS and (labels['suggested_severity']=='P0' or labels['urgency']=='critical'))
    serious=bool(flags & SERIOUS_FLAGS) or 'text_discrimination' in flags
    serious=serious or bool(labels and labels['topic']=='rider_conduct' and labels['sentiment'] in {'negative','mixed'})
    unassessed='independent_text_risk' in risk_flags(text,labels or {}) and not bool(flags & SERIOUS_FLAGS) and not critical
    if unassessed:flags.add('unassessed_text_safety')
    uncertain=bool(labels and (unassessed or labels['ambiguous'] or 'prompt_injection' in flags or (labels['suggested_severity']=='P0' and not critical)))
    if labels is None:missing.append('valid_canonical_classification')
    response=bool(labels and (response_required_by_context or labels['sentiment'] in {'negative','mixed'} or (labels['sentiment']=='neutral' and labels['response_recommended'])))
    if critical:rule='A_CRITICAL';action=Action.URGENT_ESCALATION;priority=Priority.P0
    elif serious:rule='B_SERIOUS';action=Action.PRIORITY_REVIEW;priority=Priority.P1
    elif labels is None:rule='F_CLASSIFICATION';action=Action.ABSTAIN_FOR_HUMAN;priority=Priority.P1
    elif uncertain:rule='F_UNCERTAINTY';action=Action.ABSTAIN_FOR_HUMAN;priority=Priority.P1
    elif high is True and response:rule='C_HIGH_SPREAD';action=Action.PRIORITY_REVIEW;priority=Priority.P1
    elif response:rule='D_ROUTINE';action=Action.DRAFT_FOR_REVIEW;priority=Priority.P2
    else:rule='E_MONITOR';action=Action.MONITOR;priority=Priority.P3
    ids=[rule]+modifiers;reasons.insert(0,RULES[rule])
    if labels is None and rule in {'A_CRITICAL','B_SERIOUS'}:
        ids.append('F_CLASSIFICATION');reasons.append('Safety investigation is not delayed by unavailable classification.')
    if uncertain and rule in {'A_CRITICAL','B_SERIOUS'}:ids.append('F_UNCERTAINTY');reasons.append('Unsafe uncertainty remains flagged during escalation.')
    needs_policy=rule in {'A_CRITICAL','B_SERIOUS','C_HIGH_SPREAD','D_ROUTINE'}
    needs_draft=rule in {'C_HIGH_SPREAD','D_ROUTINE'}
    status='ESCALATION_REQUIRED' if rule in {'A_CRITICAL','B_SERIOUS'} else 'AWAITING_POLICY' if needs_policy else 'ABSTAINED' if action==Action.ABSTAIN_FOR_HUMAN else 'MONITORING'
    if stage!='preliminary' and needs_policy and (not isinstance(retrieval,dict) or retrieval.get('status')!='sufficient' or not retrieval.get('evidence')):
        ids.append('F_POLICY');missing.append('sufficient_nonconflicting_policy_evidence');reasons.append(RULES['F_POLICY']);needs_draft=False;status='POLICY_UNAVAILABLE'
        if action==Action.DRAFT_FOR_REVIEW:action=Action.ABSTAIN_FOR_HUMAN;priority=Priority.P1
    if stage=='final' and needs_draft:
        validation=(generation or {}).get('validation') or {}
        if not generation or generation.get('status')!='review_required' or not generation.get('draft') or validation.get('status')!='review_required' or validation.get('issues') or not validation.get('citation_valid') or not validation.get('privacy_pass'):
            ids.append('F_DRAFT');missing.append('safe_grounded_draft');reasons.append(RULES['F_DRAFT']);needs_draft=False;status='DRAFT_BLOCKED'
            if action==Action.DRAFT_FOR_REVIEW:action=Action.ABSTAIN_FOR_HUMAN;priority=Priority.P1
        else:status='READY_FOR_HUMAN'
    retrieved_ids=sorted({e.get('policy_id') for e in (retrieval or {}).get('evidence',[]) if isinstance(e,dict) and e.get('kind')=='policy' and isinstance(e.get('policy_id'),str)})
    applicable=sorted(set(retrieved_ids)&set(POLICY_BINDINGS.get(rule,[])))
    if applicable:reasons.append('Retrieved proposed guidance for human investigation: '+', '.join(applicable)+'. This does not verify the allegation or authorize publication.')
    if 'ml_prediction' in missing:status+='__ML_UNAVAILABLE'
    return Decision(post_id=post_id,action=action,priority=priority,decision_reasons=reasons,rule_ids=list(dict.fromkeys(ids)),triggering_safety_flags=sorted(flags),retrieved_policy_ids=retrieved_ids,suggested_severity=labels['suggested_severity'] if labels else None,predicted_spread_probability=p,operational_spread_threshold=threshold,high_spread=high,requires_human_review=action!=Action.MONITOR,requires_response_draft=needs_draft,requires_policy_retrieval=needs_policy,missing_inputs=missing,decision_rule_version=VERSION,decision_rule_fingerprint=RULE_FINGERPRINT,processing_status=status)
