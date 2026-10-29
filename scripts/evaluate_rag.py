import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
p=argparse.ArgumentParser();p.add_argument('--secrets-file');p.add_argument('--limit',type=int,default=8);p.add_argument('--output');p.add_argument('--retry-transport',action='store_true');a=p.parse_args()
if a.secrets_file:
 from config import load_secrets_for_script
 load_secrets_for_script(a.secrets_file)
from src.rag.evaluate import evaluate
report,path=evaluate(a.limit,a.output,retry_transport=a.retry_transport)
print(json.dumps(report,indent=2));print('Saved evaluation:',path)
