"""Policy-query redaction helper; response checks live in guardrails.py."""
import re

PATTERNS=[
    (r'\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b','[EMAIL]'),
    (r'\b(?:\d[ -]?){13,19}\b','[PAYMENT_NUMBER]'),
    (r'(?i)\b(?:OTP|PIN|CVV|password|order\s*(?:id|number)|address|phone|mobile)\s*(?:is|:|=|#)?\s*[^\n,.!?;]{1,100}','[PRIVATE_DETAIL]'),
    (r'(?<!\w)(?:\+?\d[\d ()-]{8,}\d)(?!\w)','[PHONE_OR_ID]'),
]
def redact(text):
    for pattern,replacement in PATTERNS:
        text=re.sub(pattern,replacement,text)
    return text

