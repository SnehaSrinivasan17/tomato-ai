"""Independent conservative draft validation; no publishing or final decision engine."""
import re
from llm.storage import redact as basic_redact
from src.rag.retrieve import classification_dict

VALIDATOR_VERSION='1.2.0'

RISK=re.compile(r'cannot breathe|throat.{0,20}swell|allerg|poison|glass in|cockroach|insect|raw chicken|undercooked|vomit|contamin|tamper|stalk|hate speech|caste|threat|harass|discriminat|account.{0,15}(?:hack|compromis)|address.{0,20}(?:public|leak)|saans|saans nahi|dhamki|keeda|ज़हर|एलर्जी|सांस|धमकी|निजी|उत्पीड़न|उल्टी|कीड़ा|कांच|हिंसा',re.I)
INJECTION=re.compile(r'ignore.{0,25}(?:instruction|previous)|reveal.{0,30}(?:prompt|key|secret)|system prompt|पिछले निर्देश|instructions ignore',re.I)
PII=re.compile(r'\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b|(?<!\w)(?:\+?\d[\d ()-]{8,}\d)(?!\w)|\b(?:sk-|gsk_)[A-Za-z0-9_-]{12,}|\b\d+\s+\w+\s+(?:street|road|lane|avenue)\b',re.I)
CLAIM=re.compile(r'\b(?:refund|compensation|credit|redelivery)\b|रिफंड|मुआवज़|मुआवजा|पैसे|पैसा|वापसी|refund|paise.{0,15}wapas',re.I)
COMMIT=re.compile(r'\b(?:will|have|already|guarantee|approved|issued|processed|completed)\b|कर दिया|कर देंगे|मिलेगा|मिलेगी|आ जाएगा|हो जाएगा|करेंगे|ho gaya|kar diya|mil jayega|mil jaayega|denge|pakka|wapas kar',re.I)
FACT=re.compile(r'we.{0,30}(?:investigated|contacted|escalated|confirmed|fired)|investigation.{0,20}(?:complete|done)|food.{0,15}(?:safe|poisoned)|restaurant.{0,15}(?:at fault|responsible)|within\s+\d+|\d+\s*(?:minutes|hours|घंटे|मिनट).{0,20}(?:refund|resolve)|हमने.{0,20}(?:जांच|संपर्क|कार्रवाई)|जांच.{0,20}(?:पूरी|हो चुकी)|kar di.{0,10}(?:jaanch|action)',re.I)
SOLICIT=re.compile(r'(?:send|share|provide|post|भेजें|बताएं|bhejo|batao).{0,45}(?:otp|cvv|pin|password|address|phone|पता|फोन|पासवर्ड)',re.I)


def risk_flags(text,classification):
    labels=classification_dict(classification)
    flags=set(labels.get('safety_flags',[]))-{'prompt_injection'}
    if labels.get('urgency')=='critical' or labels.get('suggested_severity') in {'P0','P1'}:flags.add('classifier_high_severity')
    if RISK.search(text):flags.add('independent_text_risk')
    return sorted(flags)


def validate_response(draft,evidence,classification,post_text=''):
    issues=[];text=draft.draft_response;by_id={e['policy_id']:e for e in evidence}
    ids=set(draft.cited_policy_ids)
    if not ids or not ids.issubset(by_id):issues.append('invalid_or_missing_citations')
    if any(policy_id in text for policy_id in by_id):issues.append('internal_policy_identifier_in_public_draft')
    if not draft.supporting_evidence:issues.append('missing_claim_evidence')
    for support in draft.supporting_evidence:
        e=by_id.get(support.policy_id)
        if e is None or not support.quote.strip() or support.quote not in e.get('chunk_text','') or support.claim not in text or support.policy_id not in ids:issues.append('invalid_evidence_span')
    if PII.search(text) or basic_redact(text)!=text:issues.append('privacy_exposure')
    if SOLICIT.search(text) or re.search(r'(?:ओटीपी|सीवीवी|पिन|पासवर्ड|फोन नंबर|पता).{0,35}(?:भेजें|बताएं|साझा)|(?:otp|cvv|pin|password|phone|address).{0,30}(?:bhejo|batao)',text,re.I):issues.append('public_sensitive_data_request')
    if INJECTION.search(text):issues.append('instruction_leak_or_injection')
    if (CLAIM.search(text) and COMMIT.search(text)) or FACT.search(text) or re.search(r'(?:we|team|humari|hamari|हमारी).{0,50}(?:will|soon|jald|जल्द).{0,35}(?:investigat|review|resolve|contact|escalat|jaanch|जांच)',text,re.I):issues.append('unsupported_commitment_or_case_assertion')
    if re.search(r'\b(?:system|developer) prompt\b|gsk_|GROQ_API_KEY',text,re.I):issues.append('internal_information')
    if len(text)>1200:issues.append('draft_too_long')
    # Free-form semantic entailment is not claimed by a string-span check.
    flags=risk_flags(post_text,classification)
    if flags:issues.append('high_risk_human_escalation')
    return dict(validator_version=VALIDATOR_VERSION,status='blocked' if issues else 'review_required',issues=sorted(set(issues)),risk_flags=flags,citation_valid=not any(x in issues for x in ['invalid_or_missing_citations','invalid_evidence_span']),grounding_check='evidence spans and conservative commitment checks; human entailment review required',privacy_pass=not any(x in issues for x in ['privacy_exposure','public_sensitive_data_request']),requires_human_review=True,escalation_required=bool(flags),publish_allowed=False)


def sanitize_record(value):
    if isinstance(value,dict):return {k:sanitize_record(v) for k,v in value.items()}
    if isinstance(value,list):return [sanitize_record(v) for v in value]
    if isinstance(value,str):return PII.sub('[PRIVATE]',basic_redact(value))
    return value
