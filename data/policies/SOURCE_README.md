# Tomato enterprise RAG v3 — academic simulation

Created 2026-10-28. Contains 54 policy chunks, 60 response templates and 75 evaluation scenarios.

## Important
All Tomato thresholds and procedures are proposed, not real company commitments. External company practices are comparative benchmarks only. The legal register includes a government-hosted document whose operative status should be checked against current consolidated legislation. DPDP implementation is phased; do not assume all provisions commenced.

## Ingestion
Index rag_chunks.jsonl by policy_id with metadata filters status, version, severity, and owner. Treat only approved/current chunks as authoritative for deployment. Store response templates separately. Retrieve relevant policy chunks before templates. Cite policy IDs in internal trace; never reveal confidential notes publicly.

## Decision workflow
Detect language and harm -> classify P0-P3 -> check exposed PII -> retrieve current policy -> verify authenticated case state -> produce grounded draft -> block high risk -> route human review -> record audit.

## Evaluation
Use evaluation_scenarios.csv for regression tests; assert severity, escalation, privacy, no invented refunds, and policy-ID coverage. Many test rows are systematic paraphrase variants; expand with real anonymised examples before production.

## Outstanding approvals
Compensation matrix; operating hours; legal review of Indian consumer law and privacy implementation; incident contact tree; staffing; final public terms; retention periods; source recertification.
