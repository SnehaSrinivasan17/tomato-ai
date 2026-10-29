"""One cached official Groq client factory, shared by classification and drafting."""
from functools import lru_cache
@lru_cache(maxsize=4)
def shared_client(api_key,timeout=45):
    from groq import Groq
    return Groq(api_key=api_key,timeout=timeout,max_retries=0)
