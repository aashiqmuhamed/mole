# Mole corpora

Generated Mole corpora land here (this directory is gitignored except this README
and the bundled [`sample/`](sample/)). Produce one with `scripts/sim_runner.py`.
Each corpus is a set of files sharing a stem:

    <stem>.jsonl                audit-event log (what the monitors train / eval on)
    <stem>.transcripts.jsonl    full per-session transcripts (what the LLM judge reads)
    <stem>.jsonl.resume.json    day-boundary resume state (crash backstop)
    <stem>.log                  generation log

Typical pipeline (see the repo README "Quickstart"):

    # 1. generate
    python scripts/sim_runner.py --full --days 30 --attack-rate 0.05 \
        --out data/corpus/<stem>.jsonl

    # 2. label — privileged transcript judge → 4-way per-insider-day outcome
    python scripts/llm_harm_label.py \
        --transcripts data/corpus/<stem>.transcripts.jsonl \
        --out data/labels/<stem>.json --judge-model openai/gpt-4o

    # 3. grade — monitors, temporal split, R@daily-budget / pAUC / AUROC
    python scripts/auc_eval.py --audit data/corpus/<stem>.jsonl \
        --llm-truth data/labels/<stem>.json \
        --monitors random,zscore,peer_fit,facade,cascade

Use a judge model from a **different family** than the generator to avoid
self-collusion. `--resume` continues an interrupted run at the last committed day.
