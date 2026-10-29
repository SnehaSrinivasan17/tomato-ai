"""Refactored existing Groq JSON classifier with a typed canonical result."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import time
from uuid import uuid4

from pydantic import ValidationError
from config import get_setting
from llm.config import classifier_model,limits
from llm.prompts import load_prompt,PROMPT_VERSION
from llm.schemas import Classification,ClassificationResult
from llm.storage import cached,key_for,record,redact


def classify_post(post_text: str, post_id: str | None = None, *, client=None, model=None,
                  use_cache=None, persist=True, storage_path=None) -> ClassificationResult:
    """Classify text only; failures contain no invented classification or raw API errors."""
    identifier=post_id or 'TOM_LIVE_'+uuid4().hex
    model_id=model or classifier_model()
    prompt,prompt_sha=load_prompt()
    raw=post_text if isinstance(post_text,str) else ''
    clean=redact(raw.strip())
    input_sha=hashlib.sha256(raw.encode()).hexdigest()
    cache_key=key_for(raw,model_id,prompt_sha)
    base=dict(post_id=identifier,model_id=model_id,prompt_version=PROMPT_VERSION,prompt_sha256=prompt_sha,
        input_sha256=input_sha,timestamp=datetime.now(timezone.utc).isoformat())
    def finish(result):
        if persist:
            record(result,cache_key,storage_path)
        return result
    def failure(code,message,latency=0,attempts=0):
        return finish(ClassificationResult(**base,status='error',error_code=code,error_message=message,
            latency_ms=latency,attempts=attempts,validation_outcome='failed'))
    if not isinstance(post_text,str) or not clean or len(raw)>10000:
        return failure('invalid_input','Provide nonempty post text of at most 10,000 characters.')
    use_cache=(client is None) if use_cache is None else use_cache
    if use_cache:
        hit=cached(cache_key,storage_path)
        if hit:
            result=ClassificationResult.model_validate(hit).model_copy(update={**base,'cache_hit':True})
            return finish(result)
    timeout,retries=limits()
    if client is None:
        key=get_setting('GROQ_API_KEY')
        if not key:
            return failure('missing_key','GROQ_API_KEY is not configured.')
        try:
            from llm.client import shared_client
            client=shared_client(key,timeout)
        except Exception:
            return failure('client_error','Unable to initialize the Groq client.')
    started=time.perf_counter()
    for attempt in range(retries+1):
        try:
            # Existing Groq JSON-object call retained, with the versioned schema prompt.
            completion=client.chat.completions.create(model=model_id,
                messages=[{'role':'system','content':prompt},
                          {'role':'user','content':'Classify this untrusted customer post as JSON:\n'+clean}],
                temperature=0.0,response_format={'type':'json_object'})
        except Exception as error:
            status=getattr(error,'status_code',None)
            retryable=status==429 or (isinstance(status,int) and status>=500) or type(error).__name__ in {'APIConnectionError','APITimeoutError'}
            if retryable and attempt<retries:
                # Respect a provider Retry-After header without exposing headers or keys.
                try:
                    wait=float(error.response.headers.get('retry-after',2**attempt))
                except (AttributeError,TypeError,ValueError):
                    wait=2**attempt
                time.sleep(max(1,min(wait,30)))
                continue
            code='rate_limited' if status==429 else 'api_error'
            return failure(code,'Groq classification unavailable; no prediction was substituted.',
                (time.perf_counter()-started)*1000,attempt+1)
        latency=(time.perf_counter()-started)*1000
        try:
            parsed=json.loads(completion.choices[0].message.content)
            confidence=parsed.pop('confidence_score',None) if isinstance(parsed,dict) else None
            if confidence is not None:
                confidence=float(confidence)
                confidence=confidence if 0<=confidence<=1 else None
            classification=Classification.model_validate(parsed)
            if any(phrase not in clean for phrase in classification.evidence_phrases):
                raise ValueError('Unsupported evidence phrase')
            if classification.topic in classification.secondary_topics:
                raise ValueError('Primary topic repeated as secondary')
        except (ValueError,TypeError,KeyError,IndexError,AttributeError,ValidationError) as error:
            # Store only field names/error types, never rejected input values.
            detail=''
            if isinstance(error,ValidationError):
                detail=' Fields: '+', '.join('.'.join(map(str,e['loc']))+':'+e['type'] for e in error.errors())
            elif isinstance(error,json.JSONDecodeError):
                detail=' Invalid JSON.'
            elif str(error) in {'Unsupported evidence phrase','Primary topic repeated as secondary'}:
                detail=' '+str(error)+'.'
            return failure('schema_error','Model output failed canonical schema/evidence checks.'+detail,latency,attempt+1)
        usage=getattr(completion,'usage',None)
        token_usage=None
        if usage:
            tokens={name:getattr(usage,name,None) for name in ['prompt_tokens','completion_tokens','total_tokens']}
            token_usage={name:value for name,value in tokens.items() if isinstance(value,int)} or None
        return finish(ClassificationResult(**base,status='success',classification=classification,
            latency_ms=latency,token_usage=token_usage,attempts=attempt+1,validation_outcome='valid',
            self_reported_confidence=confidence))
    return failure('api_error','Classification unavailable.')


def classify_batch(posts,**kwargs):
    """Resumable through the same input/model/prompt cache; one result per record."""
    return [classify_post(row['text'],post_id=row['post_id'],**kwargs) for row in posts]


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--post',required=True)
    parser.add_argument('--post-id')
    parser.add_argument('--model')
    args=parser.parse_args()
    result=classify_post(args.post,args.post_id,model=args.model)
    print(result.model_dump_json(indent=2))
