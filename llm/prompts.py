import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PROMPT_VERSION='2.0.0'
PROMPT_PATH=ROOT/'prompts/classifier_versions/classifier_v2.0.0.md'


def load_prompt():
    text=PROMPT_PATH.read_text(encoding='utf8')
    return text,hashlib.sha256(text.encode()).hexdigest()
