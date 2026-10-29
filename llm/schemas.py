"""Canonical triage proposals, not publishing or approval decisions."""
from enum import Enum
from typing import Literal
from pydantic import BaseModel,ConfigDict,Field,StrictBool,field_validator,model_validator


class Sentiment(str,Enum):
    positive='positive'; neutral='neutral'; negative='negative'; mixed='mixed'

class Topic(str,Enum):
    food_quality='food_quality'; delivery='delivery'; app_service='app_service'; billing='billing'
    customer_support='customer_support'; general_inquiry='general_inquiry'
    privacy_security='privacy_security'; rider_conduct='rider_conduct'

class Urgency(str,Enum):
    low='low'; medium='medium'; high='high'; critical='critical'

class Severity(str,Enum):
    P0='P0'; P1='P1'; P2='P2'; P3='P3'

class Language(str,Enum):
    English='English'; Hindi='Hindi'; Hinglish='Hinglish'; other='other'

class SafetyFlag(str,Enum):
    food_safety='food_safety'; allergen='allergen'; medical_emergency='medical_emergency'
    privacy='privacy'; account_security='account_security'; harassment='harassment'
    credible_threat='credible_threat'; legal='legal'; prompt_injection='prompt_injection'

class Classification(BaseModel):
    model_config=ConfigDict(extra='forbid',protected_namespaces=())
    sentiment:Sentiment
    topic:Topic
    secondary_topics:list[Topic]
    urgency:Urgency
    suggested_severity:Severity
    language:Language
    sarcasm_detected:StrictBool
    safety_flags:list[SafetyFlag]
    response_recommended:StrictBool
    evidence_phrases:list[str]=Field(max_length=8)
    brief_reason:str=Field(min_length=1,max_length=1500)
    ambiguous:StrictBool

    @field_validator('secondary_topics','safety_flags')
    @classmethod
    def no_duplicates(cls,value):
        if len(value)!=len(set(value)):
            raise ValueError('Duplicate category')
        return value

    @model_validator(mode='after')
    def primary_not_secondary(self):
        if self.topic in self.secondary_topics:
            raise ValueError('Primary topic cannot be repeated as secondary.')
        return self


class ClassificationResult(BaseModel):
    model_config=ConfigDict(extra='forbid',protected_namespaces=())
    post_id:str
    status:Literal['success','error']
    classification:Classification|None=None
    model_id:str
    prompt_version:str
    prompt_sha256:str
    input_sha256:str
    timestamp:str
    latency_ms:float=0
    token_usage:dict[str,int]|None=None
    error_code:str|None=None
    error_message:str|None=None
    validation_outcome:str='pending'
    cache_hit:bool=False
    attempts:int=0
    self_reported_confidence:float|None=None


def legacy_candidate(record):
    """Explicit lossy mapping; never promote legacy labels to verified truth."""
    risk=record.get('risk_flag','none')
    risk={'threat_legal':'legal'}.get(risk,risk)
    flags=[risk] if risk in {'food_safety','privacy','harassment','legal'} else []
    urgency=record.get('urgency','medium')
    severity='P1' if flags or urgency=='critical' else ('P3' if urgency=='low' else 'P2')
    languages={'en':'English','hi':'Hindi','hinglish':'Hinglish'}
    def boolean(value):
        return value is True or str(value).lower() in {'true','1','yes'}
    mapped=Classification(sentiment=record['sentiment'],topic=record['topic'],secondary_topics=[],
        urgency=urgency,suggested_severity=severity,language=languages.get(record.get('language'),'other'),
        sarcasm_detected=boolean(record.get('sarcasm')),safety_flags=flags,
        response_recommended=boolean(record.get('needs_reply')),evidence_phrases=[],
        brief_reason='Provisional legacy mapping; single-risk fields cannot recover simultaneous concerns or imminent-harm evidence.',
        ambiguous=True)
    return dict(labels=mapped.model_dump(mode='json'),human_verified=False,
        legacy_human_review_advisory=boolean(record.get('human_review')),mapping_version='legacy_to_canonical_v1')
