"""Credential loading shared by Streamlit and standalone Python entry points."""

import os
from pathlib import Path
from dataclasses import dataclass, field

PROJECT_ROOT = Path(__file__).resolve().parent


def load_project_environment() -> None:
    """Load only this project's .env without replacing process variables."""
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / '.env', override=False)


def get_setting(name: str, default: str = '') -> str:
    load_project_environment()
    return _credential(name) or default


def _credential(name: str) -> str:
    # Avoid accessing st.secrets outside an active Streamlit session: scripts
    # should work without Streamlit installed or a local secrets file.
    try:
        import streamlit as st
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        from streamlit.errors import StreamlitSecretNotFoundError
    except ImportError:
        pass
    else:
        if get_script_run_ctx(suppress_warning=True) is not None:
            try:
                value = st.secrets.get(name)
            except (StreamlitSecretNotFoundError, FileNotFoundError):
                value = None
            if isinstance(value, str) and value.strip():
                return value.strip()

    return os.environ.get(name, "").strip()


@dataclass(frozen=True)
class Credentials:
    # Keep keys and the endpoint out of repr(), including in exception output.
    groq_api_key: str = field(repr=False)
    qdrant_url: str = field(repr=False)
    qdrant_api_key: str = field(repr=False)


def load_credentials() -> Credentials:
    """Use Streamlit secrets first in the app, then environment per setting.

    Standalone scripts use the process environment. Values are never logged.
    Missing-setting errors contain variable names only.
    """
    names = ("GROQ_API_KEY", "QDRANT_URL", "QDRANT_API_KEY")
    values = {name: _credential(name) for name in names}
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise ValueError("Missing required settings: " + ", ".join(missing))
    return Credentials(*(values[name] for name in names))


def load_secrets_for_script(path):
    """Explicit CLI opt-in; set process defaults without overwriting existing values."""
    import tomllib
    with Path(path).open('rb') as handle:
        values=tomllib.load(handle)
    for name,value in values.items():
        if isinstance(value,str):
            os.environ.setdefault(name,value)
