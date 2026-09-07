# Sample corpus

`sim_kimi_sample.jsonl` is a small (~2 MB) illustrative slice of a full Mole
corpus, so you can see the data shape and run the classical monitors without
generating your own. It is a slice of the **Kimi-K2.6** multiday corpus.

Format: one JSON object per line, each an `AuditEvent` (see
[`mole/audit/schema.py`](../../../mole/audit/schema.py)) — the record emitted for
every state-manager call the agents make. The monitors read exactly this stream.

```jsonc
{"event_id": "...", "ts": "2026-04-06T...Z", "account": "priya.k131",
 "account_kind": "background_llm_agent", "service": "gitlab",
 "action": "list_projects", "args": {...}, "is_malicious": false, ...}
```

What has been done to this file, and what to expect:

- **Scrubbed**: the benchmark seed-scaffold literals (which would leak oracle /
  target hints) have been stripped, using the same canonical scrubber the eval
  pipeline uses. Agent output is never altered.
- **Paper vocabulary**: keys and categorical values are migrated to the released
  terms (`account` / `account_kind`, `background_llm_agent`), including inside
  nested tool-call args, so it matches the shipped code exactly.
- **All benign**: `is_malicious` is `false` throughout. That is expected — malice
  is **not** marked at generation time; the eval pipeline labels it post-hoc
  (`scripts/eval_matrix.py` / the transcript judge). This slice is also from the
  first sim-day, before the insiders' scheduled attack-days.

So this sample demonstrates the corpus format and lets you exercise a monitor
replay end to end. It is not a labeled detection set — for that, generate a full
corpus with `scripts/sim_runner.py` and label it with `scripts/llm_harm_label.py`
(see the repo README and `data/corpus/README.md`). The full corpora and the
per-session transcripts are large and are not shipped in the repo.

Quick check:

```bash
python -c "from mole.monitors.replay import load_audit_jsonl; \
print(len(load_audit_jsonl('data/corpus/sample/sim_kimi_sample.jsonl')), 'events')"
```
