"""Recheck saved drafts without calling a model or modifying historical predictions."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.rag.guardrails import validate_response,VALIDATOR_VERSION
from src.rag.generate import Draft
from src.rag.settings import RESULTS
for directory in (RESULTS/'evaluations').glob('*'):
 report_file=directory/'report.json';checkpoint=directory/'pipeline.jsonl'
 if not report_file.exists() or not checkpoint.exists():continue
 records={}
 for line in checkpoint.read_text(encoding='utf8').splitlines():
  r=json.loads(line);records[r['post_id']]=r
 statuses=[];changed=[]
 for row in records.values():
  g=row.get('generation') or {};status=row['status']
  if g.get('status')=='review_required' and g.get('draft'):
   check=validate_response(Draft.model_validate(g['draft']),row['retrieval']['evidence'],row['classification']['classification'],row['customer_post'])
   status=check['status']
   if status!=g['status']:changed.append(dict(post_id=row['post_id'],previous_status=g['status'],current_status=status,issues=check['issues']))
  statuses.append(status)
 value=dict(validator_version=VALIDATOR_VERSION,posts=len(statuses),blocked=statuses.count('blocked'),review_required=statuses.count('review_required'),failed=sum(s in {'error','partial_failure'} for s in statuses),newly_blocked=changed,publish_allowed=0,note='Revalidation of actual saved outputs only; original predictions and validation metrics preserved. Already-blocked redacted outputs are never reopened.')
 (directory/'guardrail_revalidation.json').write_text(json.dumps(value,indent=2),encoding='utf8')
 report=json.loads(report_file.read_text(encoding='utf8'));report['current_guardrail_revalidation']=value
 report_file.write_text(json.dumps(report,indent=2),encoding='utf8')
 print(directory.name, 'current blocked:',value['blocked'],'review:',value['review_required'],'failures:',value['failed'])
