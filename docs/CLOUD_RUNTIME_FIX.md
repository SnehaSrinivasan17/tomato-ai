# Cloud RAG navigation and runtime fix

The supplied Cloud log used Python 3.11 with Streamlit 1.60.0, Starlette 1.10.0 and Uvicorn 0.54.0. It contained two separate failures.

## RAG page crash

The report name was assigned only in the Saved RAG evaluation branch, then used when opening System overview or Response quality. Report selection now runs independently for evaluation and quality views; System overview does not load evaluation files. Windows pointer paths normalize before matching. Missing, invalid or unmatched pointers fall back to an available saved report without inventing metrics.

## Static asset server error

The log also showed `Expected ASGI message http.response.start, but got http.response.body` in the Streamlit/Starlette static-file response stack. Requirements now pin Streamlit 1.50.0, whose tested server uses the earlier stack, instead of allowing an untested Cloud upgrade. This is a compatibility workaround, not a claim to have patched Starlette. Keep Python 3.11 and existing ML dependencies. No model retraining or secret changes are required.

Validation: all three RAG views, pointer fallbacks and the four main lightweight dashboard sections passed 10 focused tests using an isolated Streamlit 1.50.0 package. A separate HTTP smoke check covers health, homepage and compressed frontend assets. Cloud behavior still requires confirmation after deployment.

Deploy manually: stage src/rag/ui.py, requirements.txt, tests/test_rag_navigation.py and this document; commit; pull --rebase origin main; resolve conflicts if any; push origin main. Community Cloud should rebuild after the dependency change. Keep the app on Python 3.11. Do not commit secrets or phase7_results.

Official release reference: https://docs.streamlit.io/develop/quick-reference/release-notes/2025
