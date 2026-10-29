"""Package existing exact BGE-M3 vectors for fictional demo queries; no API calls."""
import sys,json,hashlib,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import load_secrets_for_script
from src.rag.settings import ROOT,RESULTS,EMBEDDING_MODEL,setting
from src.rag.guards import redact
from src.demo.runner import history,active_path
from src.review.store import get_decisions
from src.demo.ui import example_catalog

def main():
    load_secrets_for_script(ROOT/'.streamlit/secrets.toml')
    catalog=example_catalog();ids={p['post_id'] for _,p,_ in catalog};latest={}
    path=active_path()
    for row in history(path):
        if row['post_id'] in ids:latest.setdefault(row['post_id'],row['decision_id'])
    records=get_decisions(list(latest.values()),path)
    directory=ROOT/'data/demo/query_vectors';directory.mkdir(exist_ok=True)
    signature=EMBEDDING_MODEL+'|'+setting('BGE_MODEL_REVISION','main')+'|'+setting('BGE_M3_ENDPOINT')+'|normalized-8192-v1|'
    files={};queries={}
    for _,post,_ in catalog:
        record=records.get(latest.get(post['post_id']))
        if not record:continue
        classification=(record['pipeline'].get('classification') or {}).get('classification')
        if not classification:continue
        query=redact(post['text'])+'\n'+json.dumps({k:classification.get(k) for k in ['topic','secondary_topics','urgency','safety_flags']},ensure_ascii=False)
        name=hashlib.sha256((signature+query).encode()).hexdigest()+'.npy'
        source=RESULTS/'embedding_cache'/name
        if source.exists():
            shutil.copyfile(source,directory/name)
            files[name]=hashlib.sha256(source.read_bytes()).hexdigest();queries[name]=query
    manifest=dict(model=EMBEDDING_MODEL,revision=setting('BGE_MODEL_REVISION','main'),dimension=1024,normalization='L2, CLS pooling, max 8192 tokens',files=files,queries=queries)
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print('Packaged exact fictional demo query vectors:',len(files))
if __name__=='__main__':main()
