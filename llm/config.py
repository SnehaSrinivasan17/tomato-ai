from config import get_setting

DEFAULT_MODEL = 'openai/gpt-oss-120b'
SUPPORTED_MODELS = [DEFAULT_MODEL]


def classifier_model():
    # Preserve the original agent's project-level model configuration.
    return get_setting('GROQ_MODEL', DEFAULT_MODEL) or DEFAULT_MODEL


def limits():
    timeout=max(1,min(float(get_setting('GROQ_CLASSIFIER_TIMEOUT','45') or 45),120))
    retries=max(0,min(int(get_setting('GROQ_CLASSIFIER_RETRIES','2') or 2),3))
    return timeout,retries
