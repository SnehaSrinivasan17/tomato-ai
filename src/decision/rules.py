"""Versioned precedence, based on actual proposed policy sections, not confirmed incidents."""
import re
VERSION='1.1.0'
RULES={
 'A_CRITICAL':'Imminent danger, severe symptoms, active exposure/compromise or serious imminent threats override spread.',
 'B_SERIOUS':'Serious safety, rider misconduct, discrimination, privacy/security allegations require priority human investigation.',
 'F_CLASSIFICATION':'Missing or invalid canonical classification requires abstention unless a stronger safety rule already applies.',
 'F_UNCERTAINTY':'Ambiguity, injection or unassessable safety prevents automated drafting.',
 'C_HIGH_SPREAD':'A response-requiring post at the saved ML threshold gets priority review; spread is not proof of crisis.',
 'D_ROUTINE':'Routine unresolved complaints or response-requiring neutral inquiries require grounded drafting and human review.',
 'E_MONITOR':'Low-risk praise/information without a response requirement is monitored.',
 'F_ML_UNAVAILABLE':'Missing/unreliable ML remains unknown; safety and complaint handling continue.',
 'F_POLICY':'Missing, insufficient or contradictory evidence blocks drafting but cannot lower a safety/priority action.',
 'F_DRAFT':'Failed or unsafe generation blocks the draft and requires human handling.',
}
PRECEDENCE=['A_CRITICAL','B_SERIOUS','F_CLASSIFICATION','F_UNCERTAINTY','C_HIGH_SPREAD','D_ROUTINE','E_MONITOR']
SERIOUS_FLAGS={'food_safety','allergen','medical_emergency','privacy','account_security','harassment','credible_threat','legal'}
# These are conservative allegations/signals for investigation, never medical verification.
IMMINENT=re.compile(r'cannot breathe|can.t breathe|throat.{0,20}swell|choking|unconscious|severe.{0,15}allergic|saans.{0,15}(?:nahi|nahin)|सांस.{0,18}(?:नहीं|रुक)|गला.{0,18}सूज',re.I)
ACTIVE_EXPOSURE=re.compile(r'(?:address|phone|private data).{0,35}(?:public|posted|leaked)|posted my address|account.{0,25}(?:taken over|takeover|compromised|hacked)|took over my.{0,15}account|hacked my.{0,15}account|(?:पता|फोन).{0,30}(?:सार्वजनिक|लीक)|account.{0,20}hack',re.I)
IMMINENT_THREAT=re.compile(r'(?:hurt|kill).{0,25}(?:tonight|now|today)|(?:at|outside).{0,20}(?:door|house).{0,60}(?:threat|hurt|kill)|darwaze.{0,70}dhamki|अभी.{0,45}धमकी',re.I)
DISCRIMINATION=re.compile(r'discriminat|racist|caste.{0,20}(?:abuse|slur)|जाति.{0,25}(?:गाली|भेदभाव)',re.I)
POLICY_BINDINGS={'A_CRITICAL':['CR-101','CR-SAFE-01','CR-103','CR-PRIV-01','CR-104'],
                 'B_SERIOUS':['CR-102','CR-103','CR-104','CR-SAFE-01','SM-104','SM-MOD-01']}
