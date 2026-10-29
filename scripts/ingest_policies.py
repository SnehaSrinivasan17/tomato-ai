"""python scripts/ingest_policies.py [--secrets-file .streamlit/secrets.toml]"""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
parser=argparse.ArgumentParser();parser.add_argument('--secrets-file');parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
if args.secrets_file:
    from config import load_secrets_for_script
    load_secrets_for_script(args.secrets_file)
from src.rag.ingest import ingest,prepare
try:
    if args.prepare_only:
        chunks,audit=prepare();result={k:audit[k] for k in ['document_sections','policy_records','template_records','ingestion_blocked']}
    else:result=ingest()
except Exception as error:
    # Source/SDK messages could contain endpoint credentials: report safe type only.
    result={'status':'error','error_type':type(error).__name__,'message':'Ingestion failed; inspect configuration, policy audit, model availability and collection compatibility.'}
Path('rag_results').mkdir(exist_ok=True)
Path('rag_results/last_ingestion_attempt.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2))
if result.get('status')=='error':sys.exit(1)
