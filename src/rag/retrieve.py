"""Dense BGE-M3 retrieval with applicability, conflicts and explicit abstention."""
import json,time
from src.rag.settings import RESULTS,collection,setting
from src.rag.embeddings import BGEEncoder
from src.rag.ingest import qdrant_client
from src.rag.guards import redact


def classification_dict(value):
    if hasattr(value,'classification'): value=value.classification
    if hasattr(value,'model_dump'): value=value.model_dump(mode='json')
    return value if isinstance(value,dict) else {}


def assess(evidence,classification,post_text='',audit=None):
    flags=set(classification.get('safety_flags',[]))
    high=bool(flags-{'prompt_injection'}) or classification.get('urgency') in {'high','critical'}
    reasons=[];conflicts=[]
    if len({r['policy_id'] for r in evidence})<(2 if high else 1): reasons.append('Insufficient distinct corroborating policy passages.')
    ids={e['policy_id'] for e in evidence}
    if flags & {'food_safety','allergen','medical_emergency'} and not ids & {'CR-101','CR-102','CR-SAFE-01'}:reasons.append('No applicable food safety handling policy.')
    if flags & {'privacy','account_security'} and not ids & {'CR-103','CR-PRIV-01','KB-106'}:reasons.append('No applicable privacy/security policy.')
    if flags & {'harassment','credible_threat'} and not ids & {'CR-104','CR-101'}:reasons.append('No applicable threat-handling policy.')
    topic=classification.get('topic')
    if topic in {'billing','delivery'} and not any(i.startswith('CS-') for i in ids): reasons.append('No issue-specific customer-resolution policy.')
    for c in (audit or {}).get('conflicts',[]):
        if {c.get('rule_a'),c.get('rule_b')}.issubset(ids):
            if not c.get('precedence'): conflicts.append(c)
    if any(i in ids for i in ['SM-102','SM-SLA-01']) and any(s in post_text.lower() for s in ['deadline','response target','sla','respond within']):
        conflicts += [c for c in (audit or {}).get('conflicts',[]) if c.get('unresolved')]
    # Exceptions remain attached; no similarity score authorizes a remedy or
    # verifies an order. Relevant high-risk controls outrank routine workflows.
    return dict(status='conflict' if conflicts else 'insufficient_evidence' if reasons else 'sufficient',reasons=reasons,conflicts=conflicts,high_risk=high,simulation_only=True,requires_human_review=True)


def retrieve_policies(post_text,classification,top_k=5,*,client=None,encoder=None):
    started=time.perf_counter();result=dict(status='error',evidence=[],reasons=[],conflicts=[],requires_human_review=True)
    labels=classification_dict(classification)
    if not isinstance(post_text,str) or not post_text.strip() or not labels:
        return dict(result,reasons=['Valid customer text and classification are required.'],latency_ms=0)
    try:
        client=client or qdrant_client();encoder=encoder or BGEEncoder()
        info=client.get_collection(collection())
        if info.config.params.vectors.size!=1024:raise ValueError('Incompatible collection dimensions')
        from qdrant_client import models
        query=redact(post_text)+'\n'+json.dumps({k:labels.get(k) for k in ['topic','secondary_topics','urgency','safety_flags']},ensure_ascii=False)
        vector=encoder.encode([query],batch_size=1)[0].tolist()
        points=client.query_points(collection(),query=vector,limit=max(1,min(int(top_k),20)),with_payload=True,
            score_threshold=float(setting('RAG_MIN_SCORE','0.35')),
            query_filter=models.Filter(must=[models.FieldCondition(key='kind',match=models.MatchValue(value='policy')),models.FieldCondition(key='embedding_model',match=models.MatchValue(value='BAAI/bge-m3')),models.FieldCondition(key='ingestion_owner',match=models.MatchValue(value='tomato-policy-rag'))])).points
        evidence=[dict(p.payload,score=float(p.score),retrieval_rank=i+1) for i,p in enumerate(points)]
        audit=json.loads((RESULTS/'policy_audit.json').read_text(encoding='utf8')) if (RESULTS/'policy_audit.json').exists() else {}
        result=dict(assess(evidence,labels,post_text,audit),evidence=evidence,model='BAAI/bge-m3',collection=collection(),top_k=top_k,min_score=float(setting('RAG_MIN_SCORE','0.35')))
    except Exception:
        result['reasons']=['Policy retrieval unavailable (configuration, embedding, collection or Qdrant error); no evidence substituted.']
    result['latency_ms']=(time.perf_counter()-started)*1000
    return result
