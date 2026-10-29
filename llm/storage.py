"""Separate inference cache/events, never reference-label writes."""
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from datetime import datetime,timezone
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[1]
PREDICTIONS=ROOT/'data/llm_predictions'
DATABASE=PREDICTIONS/'inference.sqlite3'


def redact(text):
    text=re.sub(r'\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b','[EMAIL]',text)
    text=re.sub(r'(?i)\b(OTP|CVV|PIN|password)\s*(?:is|:|=)?\s*[\w@!$%#-]+',r'\1 [PRIVATE]',text)
    return re.sub(r'(?<!\w)(?:\+?\d[\d ()-]{8,}\d)(?!\w)','[PHONE_OR_PAYMENT]',text)


def key_for(text,model,prompt_sha):
    return hashlib.sha256(json.dumps([text,model,prompt_sha,'canonical-v2',0],ensure_ascii=False).encode()).hexdigest()


def connect(path=DATABASE):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(path,timeout=30)
    db.execute('CREATE TABLE IF NOT EXISTS cache (cache_key TEXT PRIMARY KEY, result_json TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS events (event_id TEXT PRIMARY KEY, post_id TEXT, model_id TEXT, prompt_version TEXT, timestamp TEXT, result_json TEXT)')
    return db


def cached(key,path=None):
    with connect(path or DATABASE) as db:
        row=db.execute('SELECT result_json FROM cache WHERE cache_key=?',(key,)).fetchone()
    return json.loads(row[0]) if row else None


def record(result,key,path=None):
    value=result.model_dump(mode='json')
    encoded=json.dumps(value,ensure_ascii=False)
    with connect(path or DATABASE) as db:
        if result.status=='success':
            db.execute('INSERT OR REPLACE INTO cache VALUES (?,?)',(key,encoded))
        db.execute('INSERT INTO events VALUES (?,?,?,?,?,?)',(uuid4().hex,result.post_id,result.model_id,
            result.prompt_version,result.timestamp,encoded))


def store_live_text(post_id,text):
    # Separate redacted live inputs; never append to the synthetic/benchmark sources.
    path=PREDICTIONS/'live_submissions.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf8') as file:
        file.write(json.dumps(dict(post_id=post_id,text=redact(text),timestamp=datetime.now(timezone.utc).isoformat()),ensure_ascii=False)+'\n')
