"""Typed deterministic decisions; approval and publication are separate."""
from enum import Enum
from typing import Literal
from pydantic import BaseModel,ConfigDict,Field

class Action(str,Enum):
    MONITOR='MONITOR';DRAFT_FOR_REVIEW='DRAFT_FOR_REVIEW';PRIORITY_REVIEW='PRIORITY_REVIEW'
    URGENT_ESCALATION='URGENT_ESCALATION';ABSTAIN_FOR_HUMAN='ABSTAIN_FOR_HUMAN'

class Priority(str,Enum):
    P0='P0';P1='P1';P2='P2';P3='P3'

class Decision(BaseModel):
    model_config=ConfigDict(extra='forbid')
    post_id:str
    action:Action
    priority:Priority
    decision_reasons:list[str]
    rule_ids:list[str]
    retrieved_policy_ids:list[str]=Field(default_factory=list)
    triggering_safety_flags:list[str]
    suggested_severity:str|None
    predicted_spread_probability:float|None=Field(default=None,ge=0,le=1)
    operational_spread_threshold:float|None=Field(default=None,gt=0,le=1)
    high_spread:bool|None=None
    requires_human_review:bool
    requires_response_draft:bool
    requires_policy_retrieval:bool
    missing_inputs:list[str]
    decision_rule_fingerprint:str|None=None
    decision_rule_version:str
    processing_status:str
    publish_allowed:Literal[False]=False
