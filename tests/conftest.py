"""Ordinary tests use isolated SQLite; real Neon checks are explicit CLI probes."""
import pytest

@pytest.fixture(autouse=True)
def isolated_storage(monkeypatch):
    monkeypatch.setenv('REVIEW_STORAGE_BACKEND','sqlite')
    from src.review import postgres
    # Streamlit secrets take precedence over environment variables inside
    # AppTest, so explicitly isolate the selector as well as its configuration.
    monkeypatch.setattr(postgres,'configured',lambda:False)
