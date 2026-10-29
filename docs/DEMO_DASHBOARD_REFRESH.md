# Demo dashboard refresh

Command Center is an operational overview. Demo & Audit follows it in navigation and owns the shared Analyze Post workflow, first-hour virality inputs, example menu and custom submissions. Fictional metadata is prefilled and explicitly distinguished from measured engagement. Submit the metadata form to apply edits before analyzing.

Results emphasize action, priority, validated draft, review status and spread probability. Sentiment, topic, urgency, evidence and other metadata follow. Internal sources and identifiers are omitted from demo presentation. Soft red/amber/green cells indicate priority and review state. The crisis page is a message selector and human-review form; review confirmation, revision checks and draft validation are retained. Saved RAG evaluation defaults to the final evaluation pointer, without loading hidden live playgrounds.

## Resource use

- Metadata forms batch input edits into one rerun.
- Read-only operational feed snapshots have a 15-second TTL and are invalidated after local analysis/review/reset. Active demo scope resolution has a 10-second TTL.
- Tables render at most 50 escaped rows; filters narrow the feed. No Styler/Jinja dependency or LargeUtf8 payload is introduced.
- Dataset source parsing and content hashes are cached; model initialization is lazy.
- Official BGE-M3 uses one CPU thread. Exact fictional-demo query vectors are packaged with SHA-256 integrity checks. Their cache keys include the complete query, model revision, endpoint identity and normalization contract. Matching queries skip local encoder loading; custom or changed queries still require the same BGE-M3 encoder or a compatible hosted endpoint. The embedding model and collection are unchanged.
- `python scripts/cache_demo_embeddings.py` packages only already-existing vectors for the fixed fictional examples; it does not call an API or substitute retrieval evidence.
- Successful pipeline snapshots retain version-aware caching. Failed results never become successful cache entries. Provider limits and blocked drafts remain visible, not fabricated.

## Credentials

An optional `GROQ_FALLBACK_API_KEY` can be set privately in Streamlit secrets or the environment. It is used only when the primary returns HTTP 401, never to bypass HTTP 429. Add this setting separately to Cloud secrets; ignored local secrets are not deployed through Git. The Groq model stays openai/gpt-oss-120b.

## Verification

Focused navigation, metadata, result-rendering, decision, cache and review tests were run. Six real live pipeline runs produced successful classifications on five examples; the sixth hit the provider rate limit. Praise monitored at P3, missing item produced a validated P2 draft, and allergy escalated at P0 despite low spread. Hinglish classified and retrieved successfully but the draft was blocked for invalid evidence spans; a bounded one-attempt correction was added while preserving validators. Another drafting request was rate-limited. These failures are not scored as successes. Raw run statuses are retained locally in ignored phase7_results/demo_refresh_live.json. There is no claim of universal live availability or measured Cloud CPU savings.

No training, dataset regeneration, model weights, thresholds or saved ML evaluation results were changed.
