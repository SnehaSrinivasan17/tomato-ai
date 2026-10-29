from pathlib import Path
from config import get_setting
ROOT=Path(__file__).resolve().parents[2]
POLICIES=ROOT/'data/policies'
RESULTS=ROOT/'rag_results'
EMBEDDING_MODEL='BAAI/bge-m3'

def setting(name,default=''):
    return get_setting(name,default)

def collection():
    return setting('QDRANT_COLLECTION','tomato_policies_v1')

def generation_model():
    return setting('GROQ_MODEL','openai/gpt-oss-120b')

COLLECTION=collection()
GENERATION_MODEL=generation_model()
