"""Policy-only draft generation using the same official client as classification."""
import hashlib,json,time,re
from typing import Literal
from pydantic import BaseModel,ConfigDict,Field,StrictBool
from config import get_setting
from llm.client import shared_client
from llm.config import limits
from llm.storage import redact
from src.rag.settings import ROOT,generation_model
from src.rag.retrieve import classification_dict
from src.rag.guardrails import validate_response

PROMPT_VERSION='1.2.0'
PROMPT_PATH=ROOT/'prompts/generation_versions/generation_v1.2.0.md'


def response_language_hint(text,labels):
    # Writing-system hint only; never replaces or changes LLM classification.
    if re.search(r'[\u0900-\u097f]',text):return 'Hindi'
    if re.search(r'\b(?:hai|mera|mere|mujhe|nahi|bhai|khana|paise|karo|aap|raha|rahi|kyun|kaise|keeda|dhamki)\b',text,re.I):return 'Hinglish'
    value=labels.get('language','English')
    return value if value in {'English','Hindi','Hinglish','other'} else 'other'


class Support(BaseModel):
    model_config=ConfigDict(extra='forbid')
    claim:str=Field(min_length=1)
    policy_id:str
    quote:str=Field(min_length=1)

class Draft(BaseModel):
    model_config=ConfigDict(extra='forbid')
    draft_response:str=Field(min_length=1,max_length=3000)
    response_language:Literal['English','Hindi','Hinglish','other']
    cited_policy_ids:list[str]
    supporting_evidence:list[Support]
    requires_human_review:Literal[True]
    escalation_recommended:StrictBool
    insufficient_policy_evidence:StrictBool
    policy_conflict_detected:StrictBool
    generation_notes:str


def generate_response(post_text,classification,retrieval,operational_context=None,spread=None,*,client=None):
    started=time.perf_counter();model=generation_model()
    result=dict(status='error',draft=None,validation=None,model_id=model,prompt_version=PROMPT_VERSION,requires_human_review=True,publish_allowed=False)
    if retrieval.get('status')!='sufficient':
        return dict(result,status='blocked',error_code='insufficient_or_conflicting_evidence',latency_ms=0)
    labels=classification_dict(classification)
    if not labels:return dict(result,error_code='missing_classification',latency_ms=0)
    prompt=PROMPT_PATH.read_text(encoding='utf-8-sig')
    language=response_language_hint(post_text,labels)
    result['requested_response_language']=language
    system_prompt=prompt+'\nServer-selected output language: '+language+'. Preserve this writing system.'
    result['prompt_sha256']=hashlib.sha256(prompt.encode()).hexdigest()
    result['system_prompt_sha256']=hashlib.sha256(system_prompt.encode()).hexdigest()
    timeout,retries=limits()
    if client is None:
        key=get_setting('GROQ_API_KEY')
        if not key:return dict(result,error_code='missing_key',latency_ms=0)
        client=shared_client(key,timeout)
    payload=dict(customer_post=redact(post_text),classification=labels,
        policies=[{k:e.get(k) for k in ['policy_id','chunk_text','source_filename','version','exceptions','approval_status']} for e in retrieval['evidence']],
        operational_context=operational_context or {},spread_context=spread)
    budget=max(256,min(int(get_setting('GROQ_DRAFT_MAX_TOKENS','1600')),4096))
    extra={'reasoning_effort':'low'} if model.startswith('openai/gpt-oss') else {}
    result['generation_config']=dict(max_completion_tokens=budget,**extra)
    for attempt in range(retries+1):
        try:
            reply=client.chat.completions.create(model=model,temperature=0,max_completion_tokens=budget,**extra,response_format={'type':'json_object'},messages=[{'role':'system','content':system_prompt},{'role':'user','content':json.dumps(payload,ensure_ascii=False)}])
            draft=Draft.model_validate_json(reply.choices[0].message.content)
            validation=validate_response(draft,retrieval['evidence'],labels,post_text)
            if draft.response_language!=language or (language in {'English','Hinglish'} and re.search(r'[\u0900-\u097f]',draft.draft_response)) or (language=='Hindi' and not re.search(r'[\u0900-\u097f]',draft.draft_response)):
                validation['status']='blocked';validation['issues'].append('response_language_or_script_mismatch')
            if draft.insufficient_policy_evidence or draft.policy_conflict_detected:
                validation['status']='blocked';validation['issues'].append('model_reports_insufficiency_or_conflict')
            usage=getattr(reply,'usage',None)
            return dict(result,status=validation['status'],draft=draft.model_dump(mode='json'),validation=validation,attempts=attempt+1,token_usage={k:getattr(usage,k,None) for k in ['prompt_tokens','completion_tokens','total_tokens']} if usage else None,latency_ms=(time.perf_counter()-started)*1000)
        except Exception as error:
            status=getattr(error,'status_code',None)
            retryable=status==429 or (isinstance(status,int) and status>=500) or type(error).__name__ in {'APIConnectionError','APITimeoutError'}
            if retryable and attempt<retries:
                try:wait=float(error.response.headers.get('retry-after',15*(attempt+1) if status==429 else 2**attempt))
                except (AttributeError,ValueError,TypeError):wait=15*(attempt+1) if status==429 else 2**attempt
                time.sleep(max(1,min(wait,30)));continue
            return dict(result,error_code='rate_limited' if status==429 else 'api_error' if status or retryable else 'schema_or_generation_error',attempts=attempt+1,latency_ms=(time.perf_counter()-started)*1000)
