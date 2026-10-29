"""Shared Groq clients; an optional private standby is used only after HTTP 401."""
from functools import lru_cache
from types import SimpleNamespace


class AuthFallbackCompletions:
    def __init__(self, primary, standby):
        self.primary=primary
        self.standby=standby
        self.expired=False

    def create(self, **kwargs):
        if self.expired:
            return self.standby.chat.completions.create(**kwargs)
        try:
            return self.primary.chat.completions.create(**kwargs)
        except Exception as error:
            if getattr(error,'status_code',None)!=401 or self.standby is None:
                raise
            self.expired=True
            return self.standby.chat.completions.create(**kwargs)


@lru_cache(maxsize=4)
def shared_client(api_key,timeout=45):
    from groq import Groq
    from config import get_setting
    primary=Groq(api_key=api_key,timeout=timeout,max_retries=0)
    fallback=get_setting('GROQ_FALLBACK_API_KEY')
    if not fallback or fallback==api_key:
        return primary
    standby=Groq(api_key=fallback,timeout=timeout,max_retries=0)
    return SimpleNamespace(chat=SimpleNamespace(completions=AuthFallbackCompletions(primary,standby)))
